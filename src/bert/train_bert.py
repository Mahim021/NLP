"""Train Model C (pretrained BERT + sequence-classification head).

    text -> BERT tokenizer -> pretrained BERT -> classification head -> Yes/No

A standard `AutoModelForSequenceClassification` setup on `bert-base-uncased`:
no layers are frozen and no architectural surgery is performed
(PROJECT_SPEC.md Sec. 7).  Training uses the shared splits, early stopping on
validation F1, and the test split is scored once with the best checkpoint.

Local CPU run (slow but works):
    python -m src.bert.train_bert --device cpu --epochs 3

Kaggle GPU run:
    python -m src.bert.train_bert --dataset-splits /kaggle/input/<ds>/splits \
        --results-dir /kaggle/working/results --models-dir /kaggle/working/models
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits  # noqa: E402
from src.common.seeding import set_seed  # noqa: E402
from src.evaluation.metrics import compute_metrics, print_metrics, save_results  # noqa: E402

DEFAULTS = {
    "model_name": "bert",
    "pretrained_model": "bert-base-uncased",
    "seed": 42,
    "splits_dir": str(ROOT / "data" / "splits"),
    "results_dir": str(ROOT / "results"),
    "models_dir": str(ROOT / "models"),
    # 99th percentile of the corpus is ~51 words, so 64 word-pieces covers
    # nearly every example while keeping the run cheap.
    "max_len": 64,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "epochs": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "patience": 2,
    "freeze_encoder": False,   # experimental choice; off by default -- see Sec. 7
    "device": "auto",
}


def encode_split(tokenizer, df: pd.DataFrame, max_len: int) -> TensorDataset:
    enc = tokenizer(list(df["text"]), truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    labels = torch.tensor(df["label_id"].to_numpy(), dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    probs, preds, trues, losses = [], [], [], []
    for input_ids, attn, yb in loader:
        input_ids, attn, yb = input_ids.to(device), attn.to(device), yb.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        if criterion is not None:
            losses.append(criterion(logits, yb).item() * len(yb))
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(p.cpu().tolist())
        preds.extend((p >= 0.5).long().cpu().tolist())
        trues.extend(yb.cpu().tolist())
    loss = float(np.sum(losses) / len(trues)) if losses else None
    return np.array(trues), np.array(preds), np.array(probs), loss


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main() -> None:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    ap = argparse.ArgumentParser(description="Train Model C (pretrained BERT).")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "bert.json")
    ap.add_argument("--splits-dir", type=Path)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--pretrained-model", type=str)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--gradient-accumulation-steps", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", type=str)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="experimental: train only the classification head")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("splits_dir", "results_dir", "models_dir", "pretrained_model", "epochs",
                "batch_size", "gradient_accumulation_steps", "learning_rate", "max_len",
                "seed", "device"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = str(val) if key.endswith("_dir") else val
    if args.freeze_encoder:
        cfg["freeze_encoder"] = True

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"Device: {device}  |  pretrained: {cfg['pretrained_model']}")

    train, val, test = load_splits(cfg["splits_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["pretrained_model"], num_labels=2).to(device)

    if cfg["freeze_encoder"]:
        # Recorded explicitly: what is frozen, so the run is never mistaken for
        # the standard fine-tuning experiment.
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
        print("NOTE: encoder frozen -- only the classification head is trained. "
              "This is NOT the standard BERT fine-tuning result.")

    train_loader = DataLoader(encode_split(tokenizer, train, cfg["max_len"]),
                              batch_size=cfg["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]))
    val_loader = DataLoader(encode_split(tokenizer, val, cfg["max_len"]),
                            batch_size=cfg["batch_size"])
    test_loader = DataLoader(encode_split(tokenizer, test, cfg["max_len"]),
                             batch_size=cfg["batch_size"])
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    accum = max(1, cfg["gradient_accumulation_steps"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)
    cfg["n_trainable_parameters"] = int(sum(p.numel() for p in params))

    model_dir = Path(cfg["models_dir"]) / cfg["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"

    history, best_f1, best_epoch, bad_epochs = [], -1.0, -1, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        optimizer.zero_grad()
        for step, (input_ids, attn, yb) in enumerate(train_loader, start=1):
            input_ids, attn, yb = input_ids.to(device), attn.to(device), yb.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = criterion(logits, yb)
            (loss / accum).backward()
            if step % accum == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total += loss.item() * len(yb)
            seen += len(yb)
        yt, yp, ypr, val_loss = evaluate(model, val_loader, device, criterion)
        m = compute_metrics(yt, yp, ypr)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                        "val_accuracy": m["accuracy"], "val_f1": m["f1"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:02d}  train_loss={total / seen:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={m['accuracy']:.4f}  val_f1={m['f1']:.4f}  "
              f"({history[-1]['seconds']}s)")

        if m["f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = m["f1"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_f1": best_f1}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    cfg["best_epoch"] = best_epoch
    cfg["best_val_f1"] = best_f1
    # Save in HuggingFace format so the demo can load it with from_pretrained().
    model.save_pretrained(model_dir / "best_hf")
    tokenizer.save_pretrained(model_dir / "best_hf")
    (model_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    metrics, preds_df = {}, None
    for split, loader, df in (("val", val_loader, val), ("test", test_loader, test)):
        yt, yp, ypr, _ = evaluate(model, loader, device)
        metrics[split] = compute_metrics(yt, yp, ypr)
        print_metrics("Model C: BERT", split, metrics[split])
        if split == "test":
            preds_df = pd.DataFrame({
                "id": df["id"], "text": df["text"], "true_label": df["label"],
                "pred_label": ["Yes" if p else "No" for p in yp],
                "prob_actionable": ypr})

    save_results(cfg["model_name"], Path(cfg["results_dir"]), metrics, cfg,
                 history=history, predictions=preds_df)
    print(f"Best checkpoint: {model_dir / 'best_hf'} (epoch {best_epoch}, val F1 {best_f1:.4f})")


if __name__ == "__main__":
    main()
