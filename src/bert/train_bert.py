"""Train Model C: pretrained BERT + classification head.

Pipeline:
    text -> BERT tokenizer -> pretrained BERT -> classifier -> Yes/No

The model uses standard BERT fine-tuning.
No BERT layers are frozen or architecturally modified.

The shared train/validation/test splits are used.
Validation F1 is used for early stopping and checkpoint selection.
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits
from src.common.seeding import set_seed
from src.evaluation.metrics import (
    compute_metrics,
    print_metrics,
    save_results,
)


DEFAULTS = {
    "model_name": "bert",
    "pretrained_model": "bert-base-uncased",
    "seed": 42,

    "splits_dir": str(ROOT / "data" / "splits"),
    "results_dir": str(ROOT / "results"),
    "models_dir": str(ROOT / "models"),

    "max_len": 64,
    "batch_size": 16,

    "epochs": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "patience": 2,

    "device": "auto",
}


def encode_split(
    tokenizer,
    df,
    max_len,
):
    """Tokenize one dataset split."""

    encoding = tokenizer(
        list(df["text"]),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    labels = torch.tensor(
        df["label_id"].to_numpy(),
        dtype=torch.long,
    )

    return TensorDataset(
        encoding["input_ids"],
        encoding["attention_mask"],
        labels,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion=None,
):
    """Evaluate BERT on one dataset split."""

    model.eval()

    probabilities = []
    predictions = []
    true_labels = []
    losses = []

    for input_ids, attention_mask, labels in loader:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        if criterion is not None:
            losses.append(
                criterion(logits, labels).item()
                * len(labels)
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
            labels.cpu().tolist()
        )

    if true_labels:
        loss = float(
            np.sum(losses)
            / len(true_labels)
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
    """Select CPU or CUDA device."""

    if choice == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(choice)


def main():

    parser = argparse.ArgumentParser(
        description="Train Model C: pretrained BERT."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bert.json",
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
        "--pretrained-model",
        type=str,
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

    # Start with default settings.
    cfg = dict(DEFAULTS)

    # Load configuration file.
    if args.config.exists():
        with open(
            args.config,
            "r",
            encoding="utf-8",
        ) as file:
            cfg.update(json.load(file))

    # Command-line arguments override config values.
    overrides = {
        "splits_dir": args.splits_dir,
        "results_dir": args.results_dir,
        "models_dir": args.models_dir,
        "pretrained_model": args.pretrained_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_len": args.max_len,
        "seed": args.seed,
        "device": args.device,
    }

    for key, value in overrides.items():

        if value is not None:

            if (
                key.endswith("_dir")
                or key == "pretrained_model"
            ):
                cfg[key] = str(value)
            else:
                cfg[key] = value

    # Reproducibility.
    set_seed(cfg["seed"])

    device = resolve_device(
        cfg["device"]
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Pretrained model: "
        f"{cfg['pretrained_model']}"
    )

    # Load the common dataset splits.
    train, val, test = load_splits(
        cfg["splits_dir"]
    )

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )

    # Load pretrained tokenizer and BERT.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["pretrained_model"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["pretrained_model"],
        num_labels=2,
    ).to(device)

    # Standard BERT fine-tuning:
    # all pretrained BERT parameters remain trainable.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    cfg["n_trainable_parameters"] = int(
        sum(
            parameter.numel()
            for parameter in trainable_parameters
        )
    )

    print(
        f"Trainable parameters: "
        f"{cfg['n_trainable_parameters']}"
    )

    # Tokenize all three splits.
    train_dataset = encode_split(
        tokenizer,
        train,
        cfg["max_len"],
    )

    val_dataset = encode_split(
        tokenizer,
        val,
        cfg["max_len"],
    )

    test_dataset = encode_split(
        tokenizer,
        test,
        cfg["max_len"],
    )

    # Data loaders.
    generator = torch.Generator().manual_seed(
        cfg["seed"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Number of optimizer updates per epoch.
    updates_per_epoch = len(train_loader)

    total_steps = (
        updates_per_epoch
        * cfg["epochs"]
    )

    warmup_steps = int(
        total_steps
        * cfg["warmup_ratio"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
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

    # Training.
    for epoch in range(
        1,
        cfg["epochs"] + 1,
    ):

        model.train()

        start_time = time.time()

        total_loss = 0.0
        seen = 0

        for input_ids, attention_mask, labels in train_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                outputs.logits,
                labels,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                trainable_parameters,
                cfg["grad_clip"],
            )

            optimizer.step()
            scheduler.step()

            total_loss += (
                loss.item()
                * len(labels)
            )

            seen += len(labels)

        train_loss = total_loss / seen

        # Validation.
        yt, yp, ypr, val_loss = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        val_metrics = compute_metrics(
            yt,
            yp,
            ypr,
        )

        seconds = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "seconds": round(seconds, 1),
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"{seconds:.1f}s"
        )

        # Save the checkpoint with the best validation F1.
        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
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
                    f"{epoch}. "
                    f"Best epoch: {best_epoch}."
                )

                break

    # Load the best checkpoint.
    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    cfg["best_epoch"] = best_epoch
    cfg["best_val_f1"] = best_f1

    # Save the best model in Hugging Face format.
    best_hf_dir = model_dir / "best_hf"

    model.save_pretrained(
        best_hf_dir
    )

    tokenizer.save_pretrained(
        best_hf_dir
    )

    # Save configuration.
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
            "Model C: BERT",
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

    # Save metrics, history, and test predictions.
    save_results(
        cfg["model_name"],
        Path(cfg["results_dir"]),
        all_metrics,
        cfg,
        history=history,
        predictions=test_predictions,
    )

    print(
        f"Best checkpoint: {best_hf_dir} "
        f"(epoch {best_epoch}, "
        f"val F1={best_f1:.4f})"
    )


if __name__ == "__main__":
    main()