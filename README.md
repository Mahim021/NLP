# MailSense — Actionable Email Detection Using NLP

**Md Ariful Alam Mahim (2107023) · Md Jubair Husain (2107029)**

MailSense answers one question about a piece of e-mail text:

> *Does this e-mail content require the recipient's action or attention?*

It is a binary classifier over the provided Enron-based dataset, with the
original labels preserved: **`Yes` = Actionable**, **`No` = Non-Actionable**.

Because the dataset consists of e-mail **sentences / snippets** rather than
structured Subject + Body e-mails, the task is described throughout as
**actionable email-content classification**. See [PROJECT_SPEC.md](PROJECT_SPEC.md),
which is the authoritative specification for this project.

## Research question

> How do a traditional TF-IDF-based classifier, an LSTM-based neural model, and a
> pretrained Transformer-based BERT model perform for actionable email-content
> classification on our dataset?

Three approaches are compared on identical data:

| | Approach | Pipeline |
|---|---|---|
| **A** | TF-IDF + Logistic Regression | text → preprocessing → TF-IDF → Logistic Regression → Yes/No |
| **B** | LSTM | text → preprocessing → tokenizer → ids → embedding → LSTM → dense → Yes/No |
| **C** | BERT | text → BERT tokenizer → pretrained BERT → classification head → Yes/No |

## Results

Test set: 548 held-out examples (290 `No`, 258 `Yes`), the same examples for all
three models. Positive class = **Actionable (`Yes`)**, so **FN = an actionable
e-mail predicted non-actionable** — the error a user actually pays for.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| A: TF-IDF + Logistic Regression | 0.8029 | 0.7799 | 0.8101 | 0.7947 | 0.8891 | 209 | 231 | 59 | 49 |
| B: LSTM | 0.8029 | 0.7737 | 0.8217 | 0.7970 | 0.8747 | 212 | 228 | 62 | 46 |
| **C: BERT** | **0.8522** | **0.8266** | **0.8682** | **0.8469** | **0.9343** | 224 | 243 | 47 | **34** |

**BERT wins on every metric** (+4.9 accuracy points, +5.0 F1 over the baseline)
and misses 15 fewer actionable e-mails — a 31% reduction in the error that
matters. The **LSTM does not beat the TF-IDF baseline**: identical accuracy and
an F1 difference of 0.002, because 2,557 short examples are not enough to learn
embeddings from scratch. The gain comes from *pretraining*, not from recurrence.

Full analysis, error inspection, cost comparison and threats to validity:
[reports/results.md](reports/results.md) · method: [reports/methodology.md](reports/methodology.md)

Generated tables and figures: `results/comparison_test.csv`,
`results/comparison_test.md`, `results/figures/`. Per-model metrics, confusion
matrices, training histories and test predictions live in `results/<model>/`.

## Repository layout

```
data/            raw dataset, cleaned corpus, saved splits (data/README.md)
notebooks/       exploration.ipynb; training/kaggle_train_all.ipynb
src/
  preprocessing/ cleaning policy + prepare_data.py (builds the splits)
  tfidf/         Model A
  lstm/          Model B (vocab, model, training)
  bert/          Model C
  evaluation/    shared metrics, plots, compare.py
  common/        seeding, labels, split loading, paths
  predict.py     the demo: text in, Actionable/Non-Actionable + confidence
configs/         tfidf.json, lstm.json, bert.json
results/         metrics, confusion matrices, histories, comparison tables
models/          checkpoints (git-ignored)
reports/         write-up of method and findings
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# CPU-only PyTorch:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproducing the experiments

Run the data preparation once; all three models then read the same splits.

```bash
python -m src.preprocessing.prepare_data          # -> data/splits/, results/data_quality_report.json
python -m src.tfidf.train_tfidf                   # Model A (seconds, CPU)
python -m src.lstm.train_lstm                     # Model B (minutes, CPU or GPU)
python -m src.bert.train_bert                     # Model C (GPU recommended)
python -m src.evaluation.compare                  # comparison table + figures
```

Every script takes `--splits-dir`, `--results-dir`, `--models-dir`, `--seed` and
its own hyper-parameters, so the same code runs unchanged on Kaggle — no path is
hard-coded. See `notebooks/training/kaggle_train_all.ipynb`.

```bash
# example: same code, Kaggle paths
python -m src.bert.train_bert \
  --splits-dir /kaggle/working/data/splits \
  --results-dir /kaggle/working/results \
  --models-dir /kaggle/working/models \
  --batch-size 16 --epochs 4 --learning-rate 2e-5 --seed 42
```

## Demo

```bash
$ python -m src.predict --model tfidf \
    "Please review the attached draft and send your comments by Friday." \
    "Thanks for the update, looks great!"

[    Actionable]  confidence=0.842  Please review the attached draft and send your comments by Friday.
[Non-Actionable]  confidence=0.812  Thanks for the update, looks great!
```

`--model` selects `tfidf`, `lstm` or `bert`; `--file` and stdin take one e-mail
text per line; `--threshold` moves the decision boundary (lowering it trades
false positives for fewer missed actionable e-mails).

## Experimental integrity

These are properties of the code, not just intentions:

- **One split, defined once.** `prepare_data.py` writes a seeded, stratified
  70/15/15 split to `data/splits/`. All three models load those exact files.
- **No test leakage.** TF-IDF is fitted on the training split only; the LSTM
  vocabulary is built from the training split only; BERT sees only training data
  during fine-tuning. Model selection (cross-validated grid search for Model A,
  early stopping for B and C) uses training/validation data only. The test set is
  scored once, at the end.
- **Labels are never invented or altered.** `label_id` is a deterministic mapping
  of the original `Yes`/`No` string (`src/common/labels.py`), and no example is
  relabelled by hand.
- **Preprocessing keeps the signal.** No stopword removal, no stripping of modal
  verbs, action verbs or temporal expressions — "please", "confirm", "by
  tomorrow" are the cues this task depends on.
- **Reproducible.** Fixed seeds, saved splits, saved configs (`models/<model>/config.json`),
  saved training histories and metrics in `results/`.
- **Every model reports** accuracy, precision, recall, F1, the confusion matrix
  and TP/TN/FP/FN — not accuracy alone.

## Scope

Out of scope by design (PROJECT_SPEC.md §14–15): meeting/submission detection,
reply generation, summarisation, sentiment, spam detection, mailbox integration,
APIs or deployment, and any extra architectures (BoW, vanilla RNN, GRU, CNN,
Word2Vec/GloVe/FastText training).
