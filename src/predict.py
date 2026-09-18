"""MailSense demo: e-mail text in, Actionable / Non-Actionable out.

    python -m src.predict --model tfidf "Please review the attached draft by Friday."
    python -m src.predict --model lstm  --file my_emails.txt
    echo "Thanks, see you there" | python -m src.predict --model bert

This is the whole user-facing product (PROJECT_SPEC.md Sec. 14): a prediction
plus its confidence.  No server, no mailbox integration, no reply generation.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.clean import clean_text  # noqa: E402


def _load_tfidf(models_dir: Path):
    import joblib
    pipe = joblib.load(models_dir / "tfidf_logreg" / "pipeline.joblib")

    def predict(texts):
        return pipe.predict_proba(texts)[:, 1]
    return predict


def _load_lstm(models_dir: Path):
    import torch
    from src.lstm.model import LSTMClassifier
    from src.lstm.vocab import PAD_ID, encode, load_vocab

    model_dir = models_dir / "lstm"
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    cfg = ckpt["config"]
    vocab = load_vocab(model_dir / "vocab.json")
    model = LSTMClassifier(vocab_size=len(vocab), embed_dim=cfg["embed_dim"],
                           hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
                           bidirectional=cfg["bidirectional"], dropout=cfg["dropout"],
                           pad_id=PAD_ID)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def predict(texts):
        ids = torch.tensor([encode(t, vocab, cfg["max_len"]) for t in texts],
                           dtype=torch.long)
        with torch.no_grad():
            return torch.softmax(model(ids), dim=1)[:, 1].tolist()
    return predict


def _load_bert(models_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = models_dir / "bert" / "best_hf"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    max_len = int(model.config.max_position_embeddings)
    max_len = min(64, max_len)

    def predict(texts):
        enc = tokenizer(list(texts), truncation=True, padding=True,
                        max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        return torch.softmax(logits, dim=1)[:, 1].tolist()
    return predict


LOADERS = {"tfidf": _load_tfidf, "lstm": _load_lstm, "bert": _load_bert}


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify e-mail text as actionable or not.")
    ap.add_argument("texts", nargs="*", help="one or more e-mail texts")
    ap.add_argument("--model", choices=list(LOADERS), default="tfidf")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--file", type=Path, help="file with one e-mail text per line")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability above which text is called Actionable")
    args = ap.parse_args()

    texts = list(args.texts)
    if args.file:
        texts += [ln.strip() for ln in args.file.read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
    if not texts and not sys.stdin.isatty():
        texts += [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    if not texts:
        ap.error("no input text: pass text arguments, --file, or pipe into stdin")

    cleaned = [clean_text(t) for t in texts]
    probs = LOADERS[args.model](args.models_dir)(cleaned)

    for original, p in zip(texts, probs):
        verdict = "Actionable" if p >= args.threshold else "Non-Actionable"
        confidence = p if p >= args.threshold else 1 - p
        print(f"[{verdict:>14}]  confidence={confidence:.3f}  {original}")


if __name__ == "__main__":
    main()
