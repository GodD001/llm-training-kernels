"""Time BPE training on a text corpus: wall time, throughput (MB/s), peak memory, longest learned tokens.

Writes benchmarks/results/<name>.json with the environment (CPU count, library versions, date).

Example:
  python -m benchmarks.bench_tokenizer --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --out-dir out/ts
"""

import argparse
import os
import sys
import time
from pathlib import Path

from benchmarks._common import save_results
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
    p.add_argument("--name", default=None, help="results file name (default: tokenizer_<input stem>)")
    args = p.parse_args()

    size_bytes = os.path.getsize(args.input)
    print(f"input: {args.input} ({size_bytes / 1e9:.2f} GB), vocab_size={args.vocab_size}")

    start = time.perf_counter()
    vocab, merges = train_bpe(
        args.input, args.vocab_size, args.special_tokens, num_workers=args.num_workers, progress=True
    )
    elapsed = time.perf_counter() - start

    peak = peak_rss_gib()
    mb_per_s = size_bytes / 1e6 / elapsed
    print(f"time:       {elapsed:.1f} s")
    print(f"throughput: {mb_per_s:.1f} MB/s")
    print(f"memory:     {peak:.2f} GiB (peak RSS, self + children)" if peak is not None else "memory: n/a")
    print(f"vocab:      {len(vocab)} tokens, merges: {len(merges)}")

    longest = sorted(vocab.values(), key=len, reverse=True)[: args.top_k]
    print(f"top-{args.top_k} longest tokens:")
    for i, tok in enumerate(longest, 1):
        print(f"  {i:2d}. len={len(tok):3d}  {tok.decode('utf-8', errors='replace')!r}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_vocab(vocab, out / "vocab.pkl")
        save_merges(merges, out / "merges.pkl")
        print(f"saved {out / 'vocab.pkl'} and {out / 'merges.pkl'}")

    result = {
        "input": Path(args.input).name,
        "input_bytes": size_bytes,
        "vocab_size": len(vocab),
        "merges": len(merges),
        "time_s": elapsed,
        "throughput_mb_per_s": mb_per_s,
        "peak_rss_gib": peak,
        "num_workers": args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1),
        "cpu_count": os.cpu_count(),
        "longest_tokens": [t.decode("utf-8", errors="replace") for t in longest],
    }
    name = args.name or f"tokenizer_{Path(args.input).stem}"
    path = save_results(name, {k: v for k, v in vars(args).items() if k != "out_dir"}, [result])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
