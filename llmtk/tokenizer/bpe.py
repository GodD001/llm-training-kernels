"""Byte-level BPE: parallel heap-based trainer and encoder/decoder."""

from __future__ import annotations

import collections
import heapq
import multiprocessing
import os
import pickle
from collections.abc import Iterable, Iterator

import regex as re
from tqdm import tqdm

# GPT-2 / tiktoken pre-tokenization pattern.
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

CHUNK_TARGET_BYTES = 50 * 1024 * 1024  # 50 MB per pre-tokenization chunk


# ---------------------------------------------------------------------------
# Training, stage 1: parallel pre-tokenization
# ---------------------------------------------------------------------------
def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as fp:
        fp.seek(start)
        chunk_bytes = fp.read(end - start)

    text = chunk_bytes.decode("utf-8", errors="replace")
    del chunk_bytes

    local_counts: dict[tuple[bytes, ...], int] = collections.defaultdict(int)

    # Split on special tokens first so no merge crosses a document boundary.
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        segments = re.split(special_pattern, text)
    else:
        segments = [text]
    del text

    for segment in segments:
        if not segment:
            continue
        for m in PAT.finditer(segment):
            token_bytes = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            local_counts[token_bytes] += 1

    return local_counts


def _find_many_boundaries(input_path: str, special_token: bytes, target_bytes: int) -> list[int]:
    """Cut the file into ~target_bytes chunks, snapping each cut to the next special token."""
    file_size = os.path.getsize(input_path)
    boundaries = [0]

    with open(input_path, "rb") as fp:
        pos = 0
        while pos < file_size:
            target = pos + target_bytes
            if target >= file_size:
                break
            fp.seek(target)
            buf = fp.read(1024 * 1024)
            idx = buf.find(special_token)
            next_pos = target if idx == -1 else target + idx
            if next_pos <= pos:
                next_pos = target
            boundaries.append(next_pos)
            pos = next_pos

    boundaries.append(file_size)
    return sorted(set(boundaries))


# ---------------------------------------------------------------------------
# Training, stage 2: heap-based merges with incremental pair-count updates
# ---------------------------------------------------------------------------
class _MaxHeapItem:
    """Heap entry ordered by (count, pair), largest first.

    heapq is a min-heap, so the comparison is inverted: higher count wins, and on a tie the
    lexicographically larger pair wins (the required tie-breaking rule).
    """

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: _MaxHeapItem) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _get_stats_and_heap(word_counts: dict):
    """Build pair counts, the pair -> words inverted index, and the initial heap."""
    stats = collections.defaultdict(int)
    pair_to_words = collections.defaultdict(set)

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += count
            pair_to_words[pair].add(word)

    heap = [_MaxHeapItem(c, p) for p, c in stats.items()]
    heapq.heapify(heap)
    return stats, pair_to_words, heap


def _update_stats_incremental(word_counts, stats, pair_to_words, heap, best_pair, new_token) -> None:
    """Apply merge best_pair -> new_token and update only the words that contain it.

    Every pair whose count changed gets a fresh heap entry; stale entries are left in the heap
    and discarded lazily when popped.
    """
    A, B = best_pair
    affected_words = list(pair_to_words.get(best_pair, set()))
    changed_pairs = set()

    for old_word in affected_words:
        count = word_counts.get(old_word, 0)
        if count == 0:
            continue

        new_word = []
        i = 0
        while i < len(old_word):
            if i < len(old_word) - 1 and old_word[i] == A and old_word[i + 1] == B:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        new_word = tuple(new_word)

        if new_word == old_word:
            continue

        # Remove the old word's pair contributions.
        del word_counts[old_word]
        for j in range(len(old_word) - 1):
            p = (old_word[j], old_word[j + 1])
            stats[p] -= count
            changed_pairs.add(p)
            if stats[p] <= 0:
                del stats[p]
            pair_to_words[p].discard(old_word)
            if not pair_to_words[p]:
                del pair_to_words[p]

        # Add the new word's pair contributions.
        word_counts[new_word] = word_counts.get(new_word, 0) + count
        for j in range(len(new_word) - 1):
            p = (new_word[j], new_word[j + 1])
            stats[p] = stats.get(p, 0) + count
            changed_pairs.add(p)
            if p not in pair_to_words:
                pair_to_words[p] = set()
            pair_to_words[p].add(new_word)

    for p in changed_pairs:
        c = stats.get(p, 0)
        if c > 0:
            heapq.heappush(heap, _MaxHeapItem(c, p))


def _pop_best_pair(heap, stats) -> tuple[bytes, bytes] | None:
    """Pop the best pair, skipping stale entries whose count no longer matches `stats`."""
    while heap:
        item = heapq.heappop(heap)
        true_count = stats.get(item.pair, 0)
        if item.count == true_count and true_count > 0:
            return item.pair
    return None


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
    progress: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer; returns (vocab id -> bytes, ordered merges).

    `num_workers` defaults to cpu_count() - 1; `progress` shows tqdm bars.
    """
    input_path = os.fspath(input_path)
    vocab = {i: bytes([i]) for i in range(256)}
    current_id = 256
    for token in special_tokens:
        vocab[current_id] = token.encode("utf-8")
        current_id += 1

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # Stage 1: parallel pre-tokenization.
    boundary_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    boundaries = _find_many_boundaries(input_path, boundary_token, CHUNK_TARGET_BYTES)
    args_list = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=False)
        if end > start
    ]

    word_counts = collections.defaultdict(int)
    with multiprocessing.Pool(processes=num_workers) as pool:
        for local_counts in tqdm(
            pool.imap_unordered(_pretokenize_chunk, args_list, chunksize=1),
            total=len(args_list),
            unit="chunk",
            disable=not progress,
        ):
            for word, cnt in local_counts.items():
                word_counts[word] += cnt

    # Stage 2: heap-based merges.
    stats, pair_to_words, heap = _get_stats_and_heap(word_counts)
    merges = []

    with tqdm(total=num_merges, unit="merge", disable=not progress) as pbar:
        while len(merges) < num_merges:
            best_pair = _pop_best_pair(heap, stats)
            if best_pair is None:
                break  # nothing left to merge

            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[current_id] = new_token
            current_id += 1

            _update_stats_incremental(word_counts, stats, pair_to_words, heap, best_pair, new_token)
            pbar.update(1)

    return vocab, merges


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------
class Tokenizer:
    """BPE tokenizer: encodes text to ids by applying merges in rank order; decodes ids to text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Append any special tokens not already in the vocab.
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self._bytes_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = tok_bytes
                self._bytes_to_id[tok_bytes] = new_id

        # merge pair -> rank (lower rank = higher priority)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(self.merges)}

        # Split regex for special tokens, longest first so overlapping tokens match correctly.
        # The capturing group keeps the delimiters in re.split output.
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(s) for s in sorted_specials]
            self._special_pat = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pat = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load from pickle files written by `save_vocab`/`save_merges`.

        Falls back to a text format if the file is not a pickle:
          vocab: one "ID\\tHEX_BYTES" line per token; merges: one "HEX_A HEX_B" line per merge.
        """
        try:
            with open(vocab_filepath, "rb") as f:
                vocab: dict[int, bytes] = pickle.load(f)
        except Exception:
            vocab = {}
            with open(vocab_filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    id_str, hex_str = line.split("\t", 1)
                    vocab[int(id_str)] = bytes.fromhex(hex_str)

        try:
            with open(merges_filepath, "rb") as f:
                merges: list[tuple[bytes, bytes]] = pickle.load(f)
        except Exception:
            merges = []
            with open(merges_filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    h1, h2 = line.split(" ", 1)
                    merges.append((bytes.fromhex(h1), bytes.fromhex(h2)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids."""
        ids: list[int] = []
        segments = self._special_pat.split(text) if self._special_pat else [text]

        for segment in segments:
            if not segment:
                continue
            seg_bytes = segment.encode("utf-8")
            if seg_bytes in self._bytes_to_id and segment in self.special_tokens:
                ids.append(self._bytes_to_id[seg_bytes])
            else:
                for match in PAT.finditer(segment):
                    ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))
        return ids

    def _encode_pretoken(self, token_bytes: bytes) -> list[int]:
        """Apply BPE merges to one pre-token and return its token ids."""
        pieces: list[bytes] = [bytes([b]) for b in token_bytes]
        if len(pieces) <= 1:
            return [self._bytes_to_id[p] for p in pieces]

        # Repeatedly apply the lowest-rank applicable merge until none applies.
        while True:
            best_rank = len(self.merges)
            best_idx = -1
            for i in range(len(pieces) - 1):
                rank = self._merge_rank.get((pieces[i], pieces[i + 1]), len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = pieces[best_idx] + pieces[best_idx + 1]
            pieces = pieces[:best_idx] + [merged] + pieces[best_idx + 2 :]

        return [self._bytes_to_id[p] for p in pieces]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle), one chunk at a time.

        Chunks are encoded independently, so a chunk boundary must not split a special token;
        line-by-line file iteration is safe.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to a string; invalid UTF-8 becomes U+FFFD."""
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")


def save_vocab(vocab: dict[int, bytes], path: str | os.PathLike) -> None:
    """Pickle a vocab dict."""
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


def save_merges(merges: list[tuple[bytes, bytes]], path: str | os.PathLike) -> None:
    """Pickle a merges list."""
    with open(path, "wb") as f:
        pickle.dump(merges, f)


def load_vocab(path: str | os.PathLike) -> dict[int, bytes]:
    """Load a pickled vocab dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_merges(path: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    """Load a pickled merges list."""
    with open(path, "rb") as f:
        return pickle.load(f)
