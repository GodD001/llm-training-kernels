"""Training instrumentation (throughput meter, divergence detector), train.py end to end, profiling math."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import llmtk.training as training
from benchmarks.profile_attention import analytic_costs
from llmtk.training import DivergenceDetector, DivergenceError, ThroughputMeter

ROOT = Path(__file__).resolve().parents[1]


# --- ThroughputMeter ------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_throughput_meter_windows_and_warmup(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(training.time, "perf_counter", clock)
    m = ThroughputMeter(tokens_per_step=100, window=4, warmup_steps=3)
    syncs, rates = [], []
    for i in range(1, 12):
        # warmup steps take 10 s each (e.g. compilation), later steps 1 s each
        clock.t += 10.0 if i <= 3 else 1.0
        syncs.append(m.should_sync())
        r = m.step()
        if r is not None:
            rates.append(r)
    assert rates == [100.0, 100.0]  # 4 steps x 100 tokens / 4 s; warmup excluded
    assert m.median == 100.0
    assert syncs == [False, False, True, False, False, False, True, False, False, False, True]


def test_throughput_meter_reset_excludes_untimed_work(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(training.time, "perf_counter", clock)
    m = ThroughputMeter(tokens_per_step=10, window=2, warmup_steps=1)
    m.step()  # warmup ends, timer starts
    clock.t += 1.0
    m.step()
    clock.t += 100.0  # e.g. evaluation
    m.reset_window()
    clock.t += 1.0
    m.step()
    clock.t += 1.0
    assert m.step() == 10.0  # 2 steps x 10 tokens / 2 s, the 100 s pause is not counted


def test_throughput_meter_rejects_bad_args():
    with pytest.raises(ValueError):
        ThroughputMeter(10, window=0)
    with pytest.raises(ValueError):
        ThroughputMeter(10, warmup_steps=0)


# --- DivergenceDetector ---------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_divergence_non_finite(bad):
    with pytest.raises(DivergenceError, match="non-finite"):
        DivergenceDetector().check(0, bad)


def test_divergence_spike_vs_running_median():
    det = DivergenceDetector(k=3.0, window=10, min_history=5)
    for s in range(20):
        det.check(s, 2.0 + 0.01 * (s % 3))
    det.check(20, 5.9)  # below 3 x median: fine
    with pytest.raises(DivergenceError, match="running median"):
        det.check(21, 6.5)


def test_divergence_ignores_spikes_before_min_history_and_when_disabled():
    det = DivergenceDetector(k=3.0, window=10, min_history=5)
    det.check(0, 1.0)
    det.check(1, 100.0)  # not enough history yet
    off = DivergenceDetector(k=0)
    for s in range(60):
        off.check(s, 1.0)
    off.check(60, 1e6)


def test_falling_loss_never_triggers():
    det = DivergenceDetector(k=1.5, window=50, min_history=10)
    for s in range(500):
        det.check(s, 10.0 * 0.99**s)


# --- train.py end to end (CPU, tiny model, random tokens) -----------------------

TINY = [
    "--synthetic", "--device", "cpu", "--vocab_size", "64", "--d_model", "32", "--num_layers", "1",
    "--num_heads", "2", "--d_ff", "64", "--context_length", "16", "--batch_size", "4",
    "--log_interval", "5", "--throughput_warmup", "2", "--eval_batches", "2", "--final_eval_batches", "2",
    "--ckpt_path", "",
]  # fmt: skip


def run_train(tmp_path, *extra):
    cmd = [sys.executable, "train.py", *TINY, "--results_dir", str(tmp_path), "--run_name", "t", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_train_end_to_end(tmp_path):
    proc = run_train(tmp_path, "--max_iters", "30", "--eval_interval", "10")
    assert proc.returncode == 0, proc.stderr
    assert "parameters: " in proc.stdout
    summary = json.loads((tmp_path / "train_t.json").read_text())
    assert summary["status"] == "completed"
    assert summary["steps_completed"] == 30
    assert summary["tokens_seen"] == 30 * 4 * 16
    assert summary["median_tokens_per_s"] > 0 and summary["throughput_windows"] >= 4
    assert summary["final_val_loss"] is not None
    # embedding + lm_head + ln_final + one block (ln1, ln2, 4 attention projections, 3 SwiGLU matrices)
    assert summary["parameters"] == 64 * 32 + 32 * 64 + 32 + (32 + 32 + 4 * 32 * 32 + 3 * 32 * 64)
    lines = (tmp_path / "train_t_metrics.jsonl").read_text().splitlines()
    assert any("val_loss" in json.loads(line) for line in lines)
    assert any("tokens_per_s" in json.loads(line) for line in lines)


def test_train_stops_on_divergence(tmp_path):
    proc = run_train(tmp_path, "--max_iters", "50", "--lr_max", "1e4", "--warmup_iters", "1", "--eval_interval", "0")
    assert proc.returncode == 1
    assert "DIVERGENCE DETECTED" in proc.stderr
    summary = json.loads((tmp_path / "train_t.json").read_text())
    assert summary["status"] == "diverged" and summary["final_val_loss"] is None


def test_train_config_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("d_model: 32\nnot_an_option: 1\n")
    proc = run_train(tmp_path, "--config", str(cfg), "--max_iters", "1")
    assert proc.returncode != 0 and "not_an_option" in proc.stderr


def test_config_file_matches_22m_model():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/tinystories_22m.yaml").read_text())
    assert cfg["batch_size"] * cfg["context_length"] * cfg["max_iters"] == 327_680_000
    from llmtk.model import TransformerLM

    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"], cfg["num_heads"],
        cfg["d_ff"], cfg["rope_theta"],
    )  # fmt: skip
    assert sum(p.numel() for p in model.parameters()) == 22_696_448


# --- analytic attention FLOPs --------------------------------------------------------


def test_softmax_flops_share_small_for_typical_head_dim():
    costs = analytic_costs(b=4, h=12, n=512, d=64, causal=True, elem_bytes=4)
    assert costs["softmax"]["flops_share"] < 0.05
    assert sum(c["flops_share"] for c in costs.values()) == pytest.approx(1.0)
    # matmuls have far higher arithmetic intensity than softmax
    assert costs["QK^T"]["flops_per_byte"] > 10 * costs["softmax"]["flops_per_byte"]
