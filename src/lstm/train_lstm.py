"""Train Model B: LSTM with pretrained Word2Vec embeddings."""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits
from src.common.seeding import set_seed
from src.evaluation.metrics import (
    compute_metrics,
    print_metrics,
    save_results,
)
from src.lstm.model import LSTMClassifier
from src.lstm.vocab import (
    PAD_ID,
    UNK_ID,
    build_vocab,
    encode,
    save_vocab,
)


DEFAULTS = {
    "model_name": "lstm",
    "seed": 42,

    "splits_dir": str(ROOT / "data" / "splits"),
    "results_dir": str(ROOT / "results"),
    "models_dir": str(ROOT / "models"),

    "embedding_path": str(ROOT / "embeddings" / "word2vec.bin"),
    "embedding_binary": True,
    "embedding_trainable": True,

    "max_len": 64,
    "min_freq": 2,
    "max_vocab": 20000,

    "hidden_dim": 128,
    "num_layers": 1,
    "bidirectional": True,
    "dropout": 0.3,

    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "grad_clip": 5.0,
    "patience": 4,

    "device": "auto",
}


def make_embedding_matrix(
    vocab: dict,
    word_vectors: KeyedVectors,
    seed: int,
) -> tuple[torch.Tensor, float]:

    vector_size = word_vectors.vector_size

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(len(vocab), vector_size),
    ).astype(np.float32)

    # Padding vector must be zero.
    matrix[PAD_ID] = 0.0

    found = 0
    candidates = 0

    for word, idx in vocab.items():
        if idx in (PAD_ID, UNK_ID):
            continue

        candidates += 1

        if word in word_vectors:
            matrix[idx] = word_vectors[word]
            found += 1

    coverage = found / candidates if candidates else 0.0

    # Deterministic unknown-token initialization.
    matrix[UNK_ID] = 0.0

    return torch.tensor(matrix), coverage


def make_loader(
    df: pd.DataFrame,
    vocab: dict,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    generator=None,
) -> DataLoader:

    x = torch.tensor(
        [
            encode(text, vocab, max_len)
            for text in df["text"]
        ],
        dtype=torch.long,
    )

    y = torch.tensor(
        df["label_id"].to_numpy(),
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):

    model.eval()

    probs = []
    preds = []
    trues = []
    losses = []

    for xb, yb in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)

        if criterion is not None:
            losses.append(
                criterion(logits, yb).item() * len(yb)
            )

        probability = torch.softmax(logits, dim=1)[:, 1]

        probs.extend(probability.cpu().tolist())
        preds.extend(
            (probability >= 0.5)
            .long()
            .cpu()
            .tolist()
        )
        trues.extend(
            yb.cpu().tolist()
        )

    loss = (
        float(np.sum(losses) / len(trues))
        if losses
        else None
    )

    return (
        np.array(trues),
        np.array(preds),
        np.array(probs),
        loss,
    )


def resolve_device(choice: str) -> torch.device:

    if choice == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(choice)


def main():

    parser = argparse.ArgumentParser(
        description="Train MailSense LSTM."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "lstm.json",
    )

    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()

    cfg = dict(DEFAULTS)

    if args.config.exists():
        cfg.update(
            json.loads(
                args.config.read_text(
                    encoding="utf-8"
                )
            )
        )

    for key in (
        "splits_dir",
        "results_dir",
        "models_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_len",
        "seed",
        "device",
    ):

        value = getattr(args, key, None)

        if value is not None:
            cfg[key] = (
                str(value)
                if key.endswith("_dir")
                else value
            )

    set_seed(cfg["seed"])

    device = resolve_device(cfg["device"])

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load common splits
    # --------------------------------------------------

    train, val, test = load_splits(
        cfg["splits_dir"]
    )

    # --------------------------------------------------
    # Build vocabulary from TRAIN only
    # --------------------------------------------------

    vocab = build_vocab(
        train["text"],
        min_freq=cfg["min_freq"],
        max_size=cfg["max_vocab"],
    )

    cfg["vocab_size"] = len(vocab)

    print(
        f"train={len(train)} "
        f"val={len(val)} "
        f"test={len(test)} "
        f"vocab={len(vocab)}"
    )

    # --------------------------------------------------
    # Load pretrained Word2Vec
    # --------------------------------------------------

    embedding_path = Path(
        cfg["embedding_path"]
    )

    if not embedding_path.exists():

        raise FileNotFoundError(
            f"Pretrained Word2Vec file not found: "
            f"{embedding_path}"
        )

    print(
        f"Loading pretrained embeddings: "
        f"{embedding_path}"
    )

    word_vectors = (
        KeyedVectors.load_word2vec_format(
            embedding_path,
            binary=cfg["embedding_binary"],
        )
    )

    print(
        f"Embedding vocabulary: "
        f"{len(word_vectors)}"
    )

    print(
        f"Embedding dimension: "
        f"{word_vectors.vector_size}"
    )

    # --------------------------------------------------
    # Create embedding matrix
    # --------------------------------------------------

    embedding_matrix, coverage = (
        make_embedding_matrix(
            vocab,
            word_vectors,
            cfg["seed"],
        )
    )

    cfg["embedding_dim"] = (
        word_vectors.vector_size
    )

    cfg["embedding_coverage"] = coverage

    print(
        f"Vocabulary coverage: "
        f"{coverage:.2%}"
    )

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    generator = (
        torch.Generator()
        .manual_seed(cfg["seed"])
    )

    train_loader = make_loader(
        train,
        vocab,
        cfg["max_len"],
        cfg["batch_size"],
        True,
        generator,
    )

    val_loader = make_loader(
        val,
        vocab,
        cfg["max_len"],
        cfg["batch_size"],
        False,
    )

    test_loader = make_loader(
        test,
        vocab,
        cfg["max_len"],
        cfg["batch_size"],
        False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = LSTMClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
        pad_id=PAD_ID,
        embedding_trainable=cfg[
            "embedding_trainable"
        ],
    ).to(device)

    cfg["n_parameters"] = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    print(
        f"Trainable parameters: "
        f"{cfg['n_parameters']:,}"
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------

    model_dir = (
        Path(cfg["models_dir"])
        / cfg["model_name"]
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = model_dir / "best.pt"

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    history = []

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(
        1,
        cfg["epochs"] + 1,
    ):

        model.train()

        start = time.time()

        total_loss = 0.0
        seen = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["grad_clip"],
            )

            optimizer.step()

            total_loss += (
                loss.item() * len(yb)
            )

            seen += len(yb)

        train_loss = (
            total_loss / seen
        )

        yt, yp, yprob, val_loss = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        metrics = compute_metrics(
            yt,
            yp,
            yprob,
        )

        elapsed = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics[
                    "accuracy"
                ],
                "val_f1": metrics["f1"],
                "seconds": round(
                    elapsed,
                    1,
                ),
            }
        )

        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} "
            f"val_f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "state_dict":
                        model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_f1": best_f1,
                },
                best_path,
            )

        else:

            bad_epochs += 1

            if bad_epochs >= cfg["patience"]:

                print(
                    f"Early stopping at epoch "
                    f"{epoch}; best epoch "
                    f"{best_epoch}."
                )

                break

    # --------------------------------------------------
    # Save final checkpoint
    # --------------------------------------------------

    torch.save(
        {
            "state_dict":
                model.state_dict(),
            "config": cfg,
        },
        model_dir / "final.pt",
    )

    # --------------------------------------------------
    # Load BEST checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    cfg["best_epoch"] = best_epoch
    cfg["best_val_f1"] = best_f1

    save_vocab(
        vocab,
        model_dir / "vocab.json",
    )

    (
        model_dir / "config.json"
    ).write_text(
        json.dumps(
            cfg,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    results = {}
    predictions = None

    for split, loader, df in (
        ("val", val_loader, val),
        ("test", test_loader, test),
    ):

        yt, yp, yprob, _ = evaluate(
            model,
            loader,
            device,
        )

        results[split] = compute_metrics(
            yt,
            yp,
            yprob,
        )

        print_metrics(
            "Model B: LSTM",
            split,
            results[split],
        )

        if split == "test":

            predictions = pd.DataFrame(
                {
                    "id": df["id"],
                    "text": df["text"],
                    "true_label":
                        df["label"],
                    "pred_label": [
                        "Yes" if p else "No"
                        for p in yp
                    ],
                    "prob_actionable":
                        yprob,
                }
            )

    save_results(
        cfg["model_name"],
        Path(cfg["results_dir"]),
        results,
        cfg,
        history=history,
        predictions=predictions,
    )

    print(
        f"Best checkpoint: {best_path}"
    )


if __name__ == "__main__":
    main()