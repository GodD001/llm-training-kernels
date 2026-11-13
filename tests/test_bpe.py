"""BPE trainer vs a naive reference, and tokenizer round-trips."""

import collections

import pytest
import regex as re

from llmtk.tokenizer import Tokenizer, load_merges, load_vocab, save_merges, save_vocab, train_bpe
from llmtk.tokenizer.bpe import PAT

EOT = "<|endoftext|>"

CORPUS = EOT.join(
    [
        "low lower lowest newer newest wider widest",
        "the cat sat on the mat. the cat ate the hat!",
        "héllo wörld — naïve café, 猫 and 犬 live here 🐈🐕",
        "numbers 123 4567 89, it's we've they'll",
        "  multiple   spaces\n\nand\ttabs  ",
    ]
    * 20
)


def naive_train_bpe(text: str, vocab_size: int, special_tokens: list[str]):
    """Textbook BPE: recount every pair after every merge; ties broken by the larger pair."""
    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode()
    parts = re.split("|".join(re.escape(s) for s in special_tokens), text) if special_tokens else [text]
    words = collections.Counter(
        tuple(bytes([b]) for b in m.group().encode()) for part in parts for m in PAT.finditer(part)
    )
    merges = []
    while len(vocab) < vocab_size:
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:], strict=False):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        new_words = collections.Counter()
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(best[0] + best[1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] += c
        words = new_words
    return vocab, merges


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("bpe") / "corpus.txt"
    path.write_text(CORPUS, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def trained(corpus_path):
    return train_bpe(corpus_path, vocab_size=400, special_tokens=[EOT], num_workers=2)


@pytest.fixture(scope="module")
def tok(trained):
    vocab, merges = trained
    return Tokenizer(vocab, merges, special_tokens=[EOT])


# --- training -------------------------------------------------------------------


def test_train_matches_naive_reference(trained):
    vocab, merges = trained
    ref_vocab, ref_merges = naive_train_bpe(CORPUS, 400, [EOT])
    assert len(merges) > 100  # the corpus runs out of pairs at 114 merges, before reaching 400
    assert merges == ref_merges
    assert vocab == ref_vocab


def test_train_vocab_layout(trained):
    vocab, merges = trained
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == EOT.encode()
    assert len(vocab) == 256 + 1 + len(merges)
    assert len(set(vocab.values())) == len(vocab)
    # no merge crosses the special-token boundary
    assert not any(b"<|" in a + b and (a + b) != EOT.encode() for a, b in merges)


def test_train_vocab_size_not_larger_than_base(corpus_path):
    vocab, merges = train_bpe(corpus_path, vocab_size=200, special_tokens=[EOT], num_workers=1)
    assert merges == [] and len(vocab) == 257


# --- encode / decode --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat.",
        "lowest newest widest, it's",
        "héllo wörld — naïve café",
        "猫と犬 🐈🐕 emoji + CJK",
        "  leading and trailing whitespace \n\n\t",
        "unseen words: xylophone quizzical",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_empty_string(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_special_token_is_single_id(tok):
    ids = tok.encode(f"the cat{EOT}the hat{EOT}")
    eot_id = tok._bytes_to_id[EOT.encode()]
    assert ids.count(eot_id) == 2
    assert tok.decode(ids) == f"the cat{EOT}the hat{EOT}"


def test_overlapping_special_tokens_prefer_longest(trained):
    vocab, merges = trained
    double = EOT + EOT
    t = Tokenizer(vocab, merges, special_tokens=[EOT, double])
    ids = t.encode(f"a{double}b{EOT}c")
    assert t._bytes_to_id[double.encode()] in ids
    assert ids.count(t._bytes_to_id[EOT.encode()]) == 1
    assert t.decode(ids) == f"a{double}b{EOT}c"


def test_special_token_text_without_special_list_is_split(trained):
    vocab, merges = trained
    t = Tokenizer(vocab, merges)
    ids = t.encode(EOT)
    assert len(ids) > 1 and t.decode(ids) == EOT


def test_merges_compress(tok):
    text = "the cat sat on the mat."
    assert len(tok.encode(text)) < len(text.encode())


def test_encode_iterable_matches_encode(tok):
    lines = ["the cat sat\n", f"on the mat{EOT}\n", "héllo 🐈\n"]
    assert list(tok.encode_iterable(lines)) == [i for line in lines for i in tok.encode(line)]


def test_decode_invalid_utf8_uses_replacement(tok):
    assert tok.decode([0xFF]) == "�"


def test_save_load_roundtrip(trained, tmp_path):
    vocab, merges = trained
    save_vocab(vocab, tmp_path / "v.pkl")
    save_merges(merges, tmp_path / "m.pkl")
    assert load_vocab(tmp_path / "v.pkl") == vocab
    assert load_merges(tmp_path / "m.pkl") == merges
    t = Tokenizer.from_files(tmp_path / "v.pkl", tmp_path / "m.pkl", special_tokens=[EOT])
    assert t.encode("the cat sat") == Tokenizer(vocab, merges, [EOT]).encode("the cat sat")


def test_parallel_dataset_encoding_matches_sequential(trained, tmp_path, monkeypatch):
    import importlib
    from pathlib import Path

    import numpy as np

    # import the script as a normal module so spawned worker processes can import it too
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    enc = importlib.import_module("encode_dataset")

    vocab, merges = trained
    t = Tokenizer(vocab, merges, special_tokens=[EOT])
    text = (CORPUS + "\n") * 3 + "last line without newline 🐈"
    src = tmp_path / "in.txt"
    src.write_text(text, encoding="utf-8")
    seq = enc.encode_dataset_to_npy(t, src, tmp_path / "seq.npy", workers=1)
    par = enc.encode_dataset_to_npy(t, src, tmp_path / "par.npy", workers=3)
    assert np.array_equal(seq, par)
    assert t.decode(par.tolist()) == text
