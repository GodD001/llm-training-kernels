"""Tokenizer experiments: compression ratio, encode throughput, and encoding a corpus to uint16 .npy.

Examples:
  python scripts/encode_dataset.py compression --vocab V.pkl --merges M.pkl --docs valid.txt
  python scripts/encode_dataset.py throughput  --vocab V.pkl --merges M.pkl --input valid.txt
  python scripts/encode_dataset.py encode      --vocab V.pkl --merges M.pkl --input train.txt --output train.npy
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

from llmtk.tokenizer import Tokenizer

DOC_SEP = "<|endoftext|>"


def sample_documents(path: str | Path, n: int = 10, seed: int = 42) -> list[str]:
    """Sample `n` documents (delimited by <|endoftext|>) from a text file."""
    text = Path(path).read_text(encoding="utf-8")
    docs = [d for d in text.split(DOC_SEP) if d.strip()]
    rng = random.Random(seed)
    return rng.sample(docs, min(n, len(docs)))


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    """Return (bytes per token, total bytes, total tokens) over `docs`."""
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens, total_bytes, total_tokens


def estimate_throughput(tokenizer: Tokenizer, path: str | Path, max_bytes: int = 10_000_000) -> float:
    """Encode the first `max_bytes` characters of a file; return bytes/second."""
    with open(path, encoding="utf-8") as f:
        text = f.read(max_bytes)
    n_bytes = len(text.encode("utf-8"))
    start = time.perf_counter()
    tokenizer.encode(text)
    return n_bytes / (time.perf_counter() - start)


def encode_dataset_to_npy(
    tokenizer: Tokenizer, input_path: str | Path, output_path: str | Path, batch: int = 1_000_000
) -> np.ndarray:
    """Stream-encode a text file to a uint16 .npy without holding all ids as Python ints."""
    chunks, buf = [], []
    with open(input_path, encoding="utf-8") as f:
        for tid in tokenizer.encode_iterable(f):
            buf.append(tid)
            if len(buf) >= batch:
                chunks.append(np.array(buf, dtype=np.uint16))
                buf = []
    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))
    arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    np.save(output_path, arr)
    print(f"{input_path} -> {output_path} ({arr.size:,} tokens)")
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["compression", "throughput", "encode"])
    p.add_argument("--vocab", required=True, help="pickled vocab from train_bpe")
    p.add_argument("--merges", required=True, help="pickled merges from train_bpe")
    p.add_argument("--special-tokens", nargs="*", default=[DOC_SEP])
    p.add_argument("--docs", help="[compression] text file to sample documents from")
    p.add_argument("--n-docs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input", help="[throughput/encode] input text file")
    p.add_argument("--max-bytes", type=int, default=10_000_000, help="[throughput] bytes to encode")
    p.add_argument("--output", help="[encode] output .npy path")
    args = p.parse_args()

    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)

    if args.command == "compression":
        if not args.docs:
            p.error("compression requires --docs")
        r, b, t = compression_ratio(tok, sample_documents(args.docs, args.n_docs, args.seed))
        print(f"compression ratio: {r:.3f} bytes/token ({b:,} bytes / {t:,} tokens)")
    elif args.command == "throughput":
        if not args.input:
            p.error("throughput requires --input")
        bps = estimate_throughput(tok, args.input, args.max_bytes)
        pile_secs = 825 * 1024**3 / bps
        print(f"throughput: {bps / 1e6:.2f} MB/s")
        print(f"Pile (825 GB) estimate: {pile_secs / 3600:.1f} h = {pile_secs / 86400:.2f} days")
    else:
        if not (args.input and args.output):
            p.error("encode requires --input and --output")
        encode_dataset_to_npy(tok, args.input, args.output)


if __name__ == "__main__":
    main()
