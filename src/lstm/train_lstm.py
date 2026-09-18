"""Train Model B: LSTM with pretrained word embeddings.

Pipeline:
    text -> tokenizer -> token IDs -> pretrained embedding
    -> BiLSTM -> dense -> Yes/No

The vocabulary is built from the training split only.
The pretrained embedding is used to initialize the embedding layer.

Validation is used for early stopping and checkpoint selection.
The test set is evaluated only after training is complete.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import KeyedVectors

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

    "max_len": 64,
    "min_freq": 2,
    "max_vocab": 20000,

    "embedding_path": str(ROOT / "embeddings" / "word2vec.bin"),
    "embedding_binary": True,
    "embedding_trainable": True,

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


def load_word2vec(path, binary=True):
    """Load pretrained Word2Vec vectors."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained Word2Vec file not found: {path}\n"
            "Place the embedding file at this path or update "
            "'embedding_path' in configs/lstm.json."
        )

    print(f"Loading pretrained embeddings from: {path}")

    vectors = KeyedVectors.load_word2vec_format(
        str(path),
        binary=binary,
    )

    print(
        f"Embedding vocabulary: {len(vectors.key_to_index)}"
    )
    print(
        f"Embedding dimension: {vectors.vector_size}"
    )

    return vectors


def create_embedding_matrix(vocab, vectors, seed=42):
    """
    Create an embedding matrix matching our project vocabulary.

    Pretrained vectors are used when a word exists in Word2Vec.
    Random vectors are used only for words missing from the
    pretrained vocabulary.
    """

    rng = np.random.default_rng(seed)

    embed_dim = vectors.vector_size

    matrix = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(len(vocab), embed_dim),
    ).astype(np.float32)

    # Padding should have a zero vector.
    matrix[PAD_ID] = 0.0

    found = 0

    for word, word_id in vocab.items():

        if word_id == PAD_ID:
            continue

        if word in vectors.key_to_index:
            matrix[word_id] = vectors[word]
            found += 1

        elif word.lower() in vectors.key_to_index:
            matrix[word_id] = vectors[word.lower()]
            found += 1

    print(
        f"Pretrained vectors found: "
        f"{found}/{len(vocab) - 1}"
    )

    return torch.tensor(matrix, dtype=torch.float32)


def make_loader(
    df,
    vocab,
    max_len,
    batch_size,
    shuffle,
    generator=None,
):
    """Convert a dataframe into a PyTorch DataLoader."""

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

    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion=None,
):
    """Evaluate the model."""

    model.eval()

    probabilities = []
    predictions = []
    true_labels = []
    losses = []

    for xb, yb in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)

        if criterion is not None:
            losses.append(
                criterion(logits, yb).item() * len(yb)
            )

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        preds = (
            probs >= 0.5
        ).long()

        probabilities.extend(
            probs.cpu().tolist()
        )

        predictions.extend(
            preds.cpu().tolist()
        )

        true_labels.extend(
            yb.cpu().tolist()
        )

    if true_labels:
        loss = float(
            np.sum(losses) / len(true_labels)
        )
    else:
        loss = None

    return (
        np.array(true_labels),
        np.array(predictions),
        np.array(probabilities),
        loss,
    )


def resolve_device(choice):
    """Select CPU or CUDA."""

    if choice == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(choice)


def main():

    parser = argparse.ArgumentParser(
        description="Train Model B: LSTM."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "lstm.json",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
    )

    parser.add_argument(
        "--embedding-path",
        type=Path,
    )

    parser.add_argument(
        "--epochs",
        type=int,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
    )

    parser.add_argument(
        "--max-len",
        type=int,
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--device",
        type=str,
    )

    args = parser.parse_args()

    cfg = dict(DEFAULTS)

    # Load configuration file.
    if args.config.exists():
        with open(
            args.config,
            "r",
            encoding="utf-8",
        ) as file:
            cfg.update(json.load(file))

    # Command-line overrides.
    overrides = {
        "splits_dir": args.splits_dir,
        "results_dir": args.results_dir,
        "models_dir": args.models_dir,
        "embedding_path": args.embedding_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_len": args.max_len,
        "seed": args.seed,
        "device": args.device,
    }

    for key, value in overrides.items():

        if value is not None:

            if key.endswith("_dir") or key == "embedding_path":
                cfg[key] = str(value)
            else:
                cfg[key] = value

    # Reproducibility.
    set_seed(cfg["seed"])

    device = resolve_device(
        cfg["device"]
    )

    print(f"Device: {device}")

    # Load common train/validation/test data.
    train, val, test = load_splits(
        cfg["splits_dir"]
    )

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )

    # Build vocabulary ONLY from training text.
    vocab = build_vocab(
        train["text"],
        min_freq=cfg["min_freq"],
        max_size=cfg["max_vocab"],
    )

    print(f"Vocabulary size: {len(vocab)}")

    # Load pretrained Word2Vec.
    vectors = load_word2vec(
        cfg["embedding_path"],
        binary=cfg["embedding_binary"],
    )

    # Create embedding matrix aligned with our vocabulary.
    embedding_matrix = create_embedding_matrix(
        vocab,
        vectors,
        seed=cfg["seed"],
    )

    cfg["vocab_size"] = len(vocab)
    cfg["embed_dim"] = embedding_matrix.shape[1]

    # Data loaders.
    generator = torch.Generator().manual_seed(
        cfg["seed"]
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

    # Create LSTM model.
    model = LSTMClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
        pad_id=PAD_ID,
        embedding_trainable=cfg["embedding_trainable"],
    ).to(device)

    cfg["n_parameters"] = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    print(
        f"Trainable parameters: "
        f"{cfg['n_parameters']}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Model directory.
    model_dir = (
        Path(cfg["models_dir"])
        / cfg["model_name"]
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = model_dir / "best.pt"

    history = []

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    # Training loop.
    for epoch in range(
        1,
        cfg["epochs"] + 1,
    ):

        model.train()

        start_time = time.time()

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

        train_loss = total_loss / seen

        # Validation.
        yt, yp, ypr, val_loss = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        metrics = compute_metrics(
            yt,
            yp,
            ypr,
        )

        epoch_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "seconds": round(epoch_time, 1),
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={metrics['accuracy']:.4f} | "
            f"val_f1={metrics['f1']:.4f}"
        )

        # Save best checkpoint based on validation F1.
        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
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
                    f"{epoch}. Best epoch: {best_epoch}."
                )

                break

    # Save final state.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
        },
        model_dir / "final.pt",
    )

    # Load best validation checkpoint.
    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    cfg["best_epoch"] = best_epoch
    cfg["best_val_f1"] = best_f1

    # Save vocabulary and configuration.
    save_vocab(
        vocab,
        model_dir / "vocab.json",
    )

    with open(
        model_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cfg,
            file,
            indent=2,
        )

    # Final evaluation.
    all_metrics = {}
    test_predictions = None

    for split_name, loader, df in (
        ("val", val_loader, val),
        ("test", test_loader, test),
    ):

        yt, yp, ypr, _ = evaluate(
            model,
            loader,
            device,
        )

        all_metrics[split_name] = compute_metrics(
            yt,
            yp,
            ypr,
        )

        print_metrics(
            "Model B: LSTM",
            split_name,
            all_metrics[split_name],
        )

        if split_name == "test":

            test_predictions = pd.DataFrame({
                "id": df["id"],
                "text": df["text"],
                "true_label": df["label"],
                "pred_label": [
                    "Yes" if prediction == 1 else "No"
                    for prediction in yp
                ],
                "prob_actionable": ypr,
            })

    # Save results.
    save_results(
        cfg["model_name"],
        Path(cfg["results_dir"]),
        all_metrics,
        cfg,
        history=history,
        predictions=test_predictions,
    )

    print(
        f"Best checkpoint: {best_path} "
        f"(epoch {best_epoch}, "
        f"val F1={best_f1:.4f})"
    )


if __name__ == "__main__":
    main()