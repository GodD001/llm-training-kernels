#!/usr/bin/env bash
# Run the measurement pipeline stage by stage; every result lands in benchmarks/results/.
#
#   bash scripts/reproduce.sh                      # GPU stages that need no dataset
#   bash scripts/reproduce.sh data tokenizer encode train
#   bash scripts/reproduce.sh all
#
# Stages:
#   tests       pytest (Triton tests need CUDA; bf16 cases need sm80+)
#   attention   naive vs PyTorch-tiled vs Triton FA-2: latency + peak memory vs sequence length
#   model       TransformerLM forward / fwd+bwd / train step, fp32 vs bf16
#   profile     per-stage attention latency vs FLOPs (+ Nsight Systems if `nsys` is installed)
#   throughput  train.py tokens/s on random tokens: fp32, fp32+compile, bf16, bf16+compile
#   data        download TinyStories V2 (train + valid) into $DATA_DIR
#   tokenizer   train the 10k BPE on TinyStories train and time it (CPU only)
#   encode      encode train/valid to uint16 token files
#   train       full 22.7M-parameter run from configs/tinystories_22m.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data}
TOK_DIR=${TOK_DIR:-$DATA_DIR/tokenizer_ts10k}
TRAIN_DTYPE=${TRAIN_DTYPE:-bf16}       # dtype for the full training run
TRAIN_COMPILE=${TRAIN_COMPILE:-1}      # 1 = torch.compile for the full training run
RESULTS=benchmarks/results
mkdir -p "$RESULTS" "$DATA_DIR"

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(tests attention model profile throughput)
[[ "${STAGES[0]}" == "all" ]] && STAGES=(tests attention model profile throughput data tokenizer encode train)

has_bf16() { python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0) else 1)"; }
DTYPES=(fp32)
if has_bf16; then DTYPES+=(bf16)
elif [[ "${STAGES[0]}" != "__stage" ]]; then echo "note: no sm80+ GPU, skipping bf16 measurements"; fi

run_stage() {
  echo; echo "================ $1 ================"
  case "$1" in
    tests)
      python -m pytest -q -rxXs 2>&1 | tee "$RESULTS/pytest_gpu.txt" ;;
    attention)
      for dt in "${DTYPES[@]}"; do
        python -m benchmarks.bench_attention --dtype "$dt" --name "attention_${dt}"
        python -m benchmarks.bench_attention --dtype "$dt" --causal --name "attention_${dt}_causal"
      done ;;
    model)
      python -m benchmarks.bench_model --size small --context-length 512 --dtypes "${DTYPES[@]}" ;;
    profile)
      for dt in "${DTYPES[@]}"; do
        python -m benchmarks.profile_attention --dtype "$dt"
      done
      if command -v nsys >/dev/null; then
        nsys profile -t cuda,nvtx -o "$RESULTS/nsys_attention" --force-overwrite true \
          python -m benchmarks.profile_attention --name profile_attention_nsys_run
        nsys stats --report nvtx_sum "$RESULTS/nsys_attention.nsys-rep" | tee "$RESULTS/nsys_attention_nvtx_sum.txt"
      else
        echo "nsys not found: skipping the Nsight Systems capture (CUDA-event timing above still ran)"
      fi ;;
    throughput)
      common=(--config configs/tinystories_22m.yaml --synthetic --max_iters 200 --log_interval 20
              --throughput_warmup 20 --eval_interval 0 --final_eval_batches 1 --ckpt_path "")
      python train.py "${common[@]}" --dtype fp32 --run_name throughput_fp32
      python train.py "${common[@]}" --dtype fp32 --compile --run_name throughput_fp32_compile
      if has_bf16; then
        python train.py "${common[@]}" --dtype bf16 --run_name throughput_bf16
        python train.py "${common[@]}" --dtype bf16 --compile --run_name throughput_bf16_compile
      fi ;;
    data)
      base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
      for f in TinyStoriesV2-GPT4-train.txt TinyStoriesV2-GPT4-valid.txt; do
        [[ -f "$DATA_DIR/$f" ]] || curl -L --fail -o "$DATA_DIR/$f" "$base/$f"
      done
      ls -la "$DATA_DIR" ;;
    tokenizer)
      python -m benchmarks.bench_tokenizer --input "$DATA_DIR/TinyStoriesV2-GPT4-train.txt" \
        --vocab-size 10000 --out-dir "$TOK_DIR" --name tokenizer_tinystories_train ;;
    encode)
      for split in train valid; do
        python scripts/encode_dataset.py encode --vocab "$TOK_DIR/vocab.pkl" --merges "$TOK_DIR/merges.pkl" \
          --input "$DATA_DIR/TinyStoriesV2-GPT4-$split.txt" --output "$DATA_DIR/ts_$split.npy" \
          --workers "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
        # train.py memory-maps a flat uint16 file; np.save adds a .npy header, so write the raw array too.
        python -c "import numpy as np, sys; np.load(sys.argv[1], mmap_mode='r').tofile(sys.argv[2])" \
          "$DATA_DIR/ts_$split.npy" "$DATA_DIR/ts_$split.bin"
      done
      python scripts/encode_dataset.py compression --vocab "$TOK_DIR/vocab.pkl" --merges "$TOK_DIR/merges.pkl" \
        --docs "$DATA_DIR/TinyStoriesV2-GPT4-valid.txt" ;;
    train)
      flags=(--dtype "$TRAIN_DTYPE"); [[ "$TRAIN_COMPILE" == 1 ]] && flags+=(--compile)
      python train.py --config configs/tinystories_22m.yaml "${flags[@]}" \
        --train_path "$DATA_DIR/ts_train.bin" --val_path "$DATA_DIR/ts_valid.bin" \
        --ckpt_path "$DATA_DIR/ts22m.pt" --run_name tinystories_22m 2>&1 | tee "$RESULTS/train_tinystories_22m.log" ;;
    *) echo "unknown stage: $1"; exit 2 ;;
  esac
}

# Each stage runs in its own process so `set -e` stops only that stage, not the whole pipeline.
if [[ "${STAGES[0]}" == "__stage" ]]; then run_stage "${STAGES[1]}"; exit; fi

failed=()
for s in "${STAGES[@]}"; do bash "$0" __stage "$s" || failed+=("$s"); done
echo
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED stages: ${failed[*]}  (results of the other stages are in $RESULTS/)"; exit 1
fi
echo "all stages done. results in $RESULTS/"
