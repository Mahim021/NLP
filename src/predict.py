"""MailSense demo: classify e-mail text as actionable or non-actionable."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.clean import clean_text  # noqa: E402


def _load_tfidf(models_dir: Path):
    import joblib

    pipe = joblib.load(
        models_dir / "tfidf_logreg" / "pipeline.joblib"
    )

    def predict(texts):
        return pipe.predict_proba(texts)[:, 1].tolist()

    return predict


def _load_lstm(models_dir: Path):
    import json

    import torch
    from gensim.models import KeyedVectors

    from src.lstm.model import LSTMClassifier
    from src.lstm.vocab import PAD_ID, encode, load_vocab

    model_dir = models_dir / "lstm"

    checkpoint = torch.load(
        model_dir / "best.pt",
        map_location="cpu",
    )

    cfg = checkpoint["config"]

    vocab = load_vocab(model_dir / "vocab.json")

    embedding_path = Path(cfg["embedding_path"])
    if not embedding_path.is_absolute():
        embedding_path = ROOT / embedding_path

    vectors = KeyedVectors.load_word2vec_format(
        embedding_path,
        binary=cfg.get("embedding_binary", True),
    )

    embedding_matrix = torch.zeros(
        len(vocab),
        vectors.vector_size,
        dtype=torch.float32,
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))

    for word, idx in vocab.items():
        if idx == PAD_ID:
            continue

        if word in vectors:
            embedding_matrix[idx] = torch.tensor(
                vectors[word],
                dtype=torch.float32,
            )
        elif word.lower() in vectors:
            embedding_matrix[idx] = torch.tensor(
                vectors[word.lower()],
                dtype=torch.float32,
            )
        else:
            embedding_matrix[idx] = torch.randn(
                vectors.vector_size,
                generator=generator,
            ) * 0.05

    model = LSTMClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
        pad_id=PAD_ID,
        embedding_trainable=cfg.get("embedding_trainable", True),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def predict(texts):
        ids = torch.tensor(
            [
                encode(text, vocab, cfg["max_len"])
                for text in texts
            ],
            dtype=torch.long,
        )

        with torch.no_grad():
            probabilities = torch.softmax(model(ids), dim=1)[:, 1]

        return probabilities.tolist()

    return predict


def _load_bert(models_dir: Path):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    path = models_dir / "bert" / "best_hf"

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    model.eval()

    max_len = min(
        64,
        int(model.config.max_position_embeddings),
    )

    def predict(texts):
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**encodings).logits

        return torch.softmax(logits, dim=1)[:, 1].tolist()

    return predict


LOADERS = {
    "tfidf": _load_tfidf,
    "lstm": _load_lstm,
    "bert": _load_bert,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify e-mail text as actionable or non-actionable."
    )

    parser.add_argument(
        "texts",
        nargs="*",
        help="one or more e-mail texts",
    )

    parser.add_argument(
        "--model",
        choices=list(LOADERS),
        default="tfidf",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="file with one e-mail text per line",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability above which text is called Actionable",
    )

    args = parser.parse_args()

    texts = list(args.texts)

    if args.file:
        texts += [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not texts and not sys.stdin.isatty():
        texts += [
            line.strip()
            for line in sys.stdin.read().splitlines()
            if line.strip()
        ]

    if not texts:
        parser.error(
            "no input text: pass text arguments, --file, or pipe into stdin"
        )

    cleaned = [
        clean_text(text)
        for text in texts
    ]

    probabilities = LOADERS[args.model](args.models_dir)(cleaned)

    for original, probability in zip(texts, probabilities):
        verdict = (
            "Actionable"
            if probability >= args.threshold
            else "Non-Actionable"
        )

        confidence = (
            probability
            if probability >= args.threshold
            else 1 - probability
        )

        print(
            f"[{verdict:>14}]  "
            f"confidence={confidence:.3f}  "
            f"{original}"
        )


if __name__ == "__main__":
    main()