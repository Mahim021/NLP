"""Train Model B (LSTM) on the shared MailSense splits.

    text -> preprocessing -> tokenizer -> ids -> embedding -> LSTM -> dense -> Yes/No

The vocabulary is built from the training split only.  The validation split
drives early stopping and checkpoint selection; the test split is scored once,
at the end, with the best checkpoint.

    python -m src.lstm.train_lstm --config configs/lstm.json --epochs 20
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
from src.lstm.model import LSTMClassifier  # noqa: E402
from src.lstm.vocab import PAD_ID, build_vocab, encode, save_vocab  # noqa: E402

DEFAULTS = {
    "model_name": "lstm",
    "seed": 42,
    "splits_dir": str(ROOT / "data" / "splits"),
    "results_dir": str(ROOT / "results"),
    "models_dir": str(ROOT / "models"),
    "max_len": 64,
    "min_freq": 2,
    "max_vocab": 20000,
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "bidirectional": True,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 5.0,
    "patience": 4,
    "device": "auto",
}


def make_loader(df: pd.DataFrame, vocab: dict, max_len: int, batch_size: int,
                shuffle: bool, generator=None) -> DataLoader:
    x = torch.tensor([encode(t, vocab, max_len) for t in df["text"]], dtype=torch.long)
    y = torch.tensor(df["label_id"].to_numpy(), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=generator)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    probs, preds, trues, losses = [], [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
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
    ap = argparse.ArgumentParser(description="Train Model B (LSTM).")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "lstm.json")
    ap.add_argument("--splits-dir", type=Path)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", type=str)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("splits_dir", "results_dir", "models_dir", "epochs", "batch_size",
                "learning_rate", "max_len", "seed", "device"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = str(val) if key.endswith("_dir") else val

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"Device: {device}")

    train, val, test = load_splits(cfg["splits_dir"])
    vocab = build_vocab(train["text"], min_freq=cfg["min_freq"], max_size=cfg["max_vocab"])
    cfg["vocab_size"] = len(vocab)
    print(f"train={len(train)} val={len(val)} test={len(test)} vocab={len(vocab)}")

    gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader = make_loader(train, vocab, cfg["max_len"], cfg["batch_size"], True, gen)
    val_loader = make_loader(val, vocab, cfg["max_len"], cfg["batch_size"], False)
    test_loader = make_loader(test, vocab, cfg["max_len"], cfg["batch_size"], False)

    model = LSTMClassifier(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"], pad_id=PAD_ID).to(device)
    cfg["n_parameters"] = int(sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])

    model_dir = Path(cfg["models_dir"]) / cfg["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"

    history, best_f1, best_epoch, bad_epochs = [], -1.0, -1, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total += loss.item() * len(yb)
            seen += len(yb)
        yt, yp, ypr, val_loss = evaluate(model, val_loader, device, criterion)
        m = compute_metrics(yt, yp, ypr)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                        "val_accuracy": m["accuracy"], "val_f1": m["f1"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:02d}  train_loss={total / seen:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={m['accuracy']:.4f}  val_f1={m['f1']:.4f}")

        if m["f1"] > best_f1:                      # checkpoint on validation F1
            best_f1, best_epoch, bad_epochs = m["f1"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_f1": best_f1}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:      # early stopping
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    torch.save({"state_dict": model.state_dict(), "config": cfg}, model_dir / "final.pt")
    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    cfg["best_epoch"] = best_epoch
    cfg["best_val_f1"] = best_f1
    save_vocab(vocab, model_dir / "vocab.json")
    (model_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    metrics, preds_df = {}, None
    for split, loader, df in (("val", val_loader, val), ("test", test_loader, test)):
        yt, yp, ypr, _ = evaluate(model, loader, device)
        metrics[split] = compute_metrics(yt, yp, ypr)
        print_metrics("Model B: LSTM", split, metrics[split])
        if split == "test":
            preds_df = pd.DataFrame({
                "id": df["id"], "text": df["text"], "true_label": df["label"],
                "pred_label": ["Yes" if p else "No" for p in yp],
                "prob_actionable": ypr})

    save_results(cfg["model_name"], Path(cfg["results_dir"]), metrics, cfg,
                 history=history, predictions=preds_df)
    print(f"Best checkpoint: {best_path} (epoch {best_epoch}, val F1 {best_f1:.4f})")


if __name__ == "__main__":
    main()
