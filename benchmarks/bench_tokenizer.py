"""Time BPE training on a text corpus and report peak memory and the longest learned tokens.

Example:
  python benchmarks/bench_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --out-dir out/ts
"""

import argparse
import os
import sys
import time
from pathlib import Path

from llmtk.tokenizer import save_merges, save_vocab, train_bpe


def peak_rss_gib() -> float | None:
    """Peak resident memory of this process plus finished child processes, in GiB (Unix only)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return usage / 1024**3 if sys.platform == "darwin" else usage / 1024**2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="training text file")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="if set, save vocab.pkl / merges.pkl here")
    p.add_argument("--top-k", type=int, default=10, help="how many longest tokens to print")
    args = p.parse_args()

    size_gb = os.path.getsize(args.input) / 1e9
    print(f"input: {args.input} ({size_gb:.2f} GB), vocab_size={args.vocab_size}")

    start = time.perf_counter()
    vocab, merges = train_bpe(
        args.input, args.vocab_size, args.special_tokens, num_workers=args.num_workers, progress=True
    )
    elapsed = time.perf_counter() - start

    peak = peak_rss_gib()
    print(f"time:   {elapsed:.1f} s")
    print(f"memory: {peak:.2f} GiB (peak RSS, self + children)" if peak is not None else "memory: n/a")
    print(f"vocab:  {len(vocab)} tokens, merges: {len(merges)}")

    print(f"top-{args.top_k} longest tokens:")
    for i, tok in enumerate(sorted(vocab.values(), key=len, reverse=True)[: args.top_k], 1):
        print(f"  {i:2d}. len={len(tok):3d}  {tok.decode('utf-8', errors='replace')!r}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_vocab(vocab, out / "vocab.pkl")
        save_merges(merges, out / "merges.pkl")
        print(f"saved {out / 'vocab.pkl'} and {out / 'merges.pkl'}")


if __name__ == "__main__":
    main()
