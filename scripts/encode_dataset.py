"""Tokenizer experiments: compression ratio, encode throughput, and encoding a corpus to uint16 .npy.

Examples:
  python scripts/encode_dataset.py compression --vocab V.pkl --merges M.pkl --docs valid.txt
  python scripts/encode_dataset.py throughput  --vocab V.pkl --merges M.pkl --input valid.txt
  python scripts/encode_dataset.py encode      --vocab V.pkl --merges M.pkl --input train.txt --output train.npy
"""

import argparse
import io
import multiprocessing
import os
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


def _encode_lines(tokenizer: Tokenizer, lines, batch: int = 1_000_000) -> np.ndarray:
    """Encode an iterable of lines (like a text file handle) to a uint16 array."""
    chunks, buf = [], []
    for tid in tokenizer.encode_iterable(lines):
        buf.append(tid)
        if len(buf) >= batch:
            chunks.append(np.array(buf, dtype=np.uint16))
            buf = []
    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)


_WORKER_TOKENIZER: Tokenizer | None = None


def _init_worker(tokenizer: Tokenizer) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _encode_byte_range(args: tuple[str, int, int]) -> np.ndarray:
    """Encode bytes [start, end) of a file line by line, exactly as iterating the file would."""
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    # StringIO with newline=None splits and translates newlines like a text-mode file handle.
    return _encode_lines(_WORKER_TOKENIZER, io.StringIO(text, newline=None))


def _line_aligned_ranges(path: str | Path, n_chunks: int) -> list[tuple[int, int]]:
    """Split a file into ~n_chunks byte ranges that each end right after a newline."""
    size = os.path.getsize(path)
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, n_chunks):
            f.seek(max(cuts[-1], size * i // n_chunks))
            f.readline()  # advance to the start of the next line
            pos = f.tell()
            if pos >= size:
                break
            if pos > cuts[-1]:
                cuts.append(pos)
    cuts.append(size)
    return list(zip(cuts[:-1], cuts[1:], strict=True))


def encode_dataset_to_npy(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    batch: int = 1_000_000,
    workers: int = 1,
) -> np.ndarray:
    """Encode a text file line by line to a uint16 .npy.

    With workers > 1 the file is split into line-aligned byte ranges that are encoded in
    parallel and concatenated in order; the output is identical to the single-process path.
    """
    if workers <= 1:
        with open(input_path, encoding="utf-8") as f:
            arr = _encode_lines(tokenizer, f, batch)
    else:
        ranges = _line_aligned_ranges(input_path, workers * 8)
        jobs = [(str(input_path), a, b) for a, b in ranges]
        with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(tokenizer,)) as pool:
            parts = pool.map(_encode_byte_range, jobs, chunksize=1)
        arr = np.concatenate(parts) if parts else np.array([], dtype=np.uint16)
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
    p.add_argument("--workers", type=int, default=1, help="[encode] parallel processes (output is identical)")
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
        encode_dataset_to_npy(tok, args.input, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
