"""Word vocabulary and tokeniser for the LSTM.

The vocabulary is built from the TRAINING split only.  Validation and test
text is encoded with that fixed vocabulary; unseen words become <unk>, so no
test information leaks into the model's vocabulary (PROJECT_SPEC.md Sec. 8).
"""
import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# Keep words, the dataset's placeholder tokens ($LINK/$EMAIL/$NUM) and the
# punctuation that carries request/question information ("?" especially).
_TOKEN_RE = re.compile(r"\$[A-Z]+|[a-z0-9']+|[?!.,;:]")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenisation; no stopword or verb removal."""
    return _TOKEN_RE.findall(_protect(text))


def _protect(text: str) -> str:
    out = text
    for tok in ("$LINK", "$EMAIL", "$NUM"):
        out = out.replace(tok, f" {tok} ")
    # lower-case everything except the placeholder tokens
    parts = []
    for piece in out.split(" "):
        parts.append(piece if piece in ("$LINK", "$EMAIL", "$NUM") else piece.lower())
    return " ".join(parts)


def build_vocab(texts, min_freq: int = 2, max_size: int | None = 20000) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    words = [w for w, c in counter.most_common() if c >= min_freq]
    if max_size is not None:
        words = words[: max_size - 2]
    vocab = {PAD: PAD_ID, UNK: UNK_ID}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def save_vocab(vocab: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")


def load_vocab(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
