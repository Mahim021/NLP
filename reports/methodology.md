# MailSense — Methodology

Md Ariful Alam Mahim (2107023) · Md Jubair Husain (2107029)

This document records *how* the experiment was run and *why* each choice was
made. Results are reported separately in [results.md](results.md). The
authoritative scope is [PROJECT_SPEC.md](../PROJECT_SPEC.md).

## 1. Task

Binary classification of e-mail content:

> Does this e-mail content require the recipient's action or attention?

`Yes` = Actionable (positive class), `No` = Non-Actionable. No other intent
classes exist in this project, and the task is not reframed as meeting
detection, submission detection, summarisation, sentiment or spam detection.

## 2. Data

The provided Enron-based dataset, `data/Ask0729-fixed.txt`: 3,657 tab-separated
`label<TAB>text` lines, 1,719 `Yes` and 1,938 `No`, no malformed rows and no
missing values.

The examples are **sentences and snippets**, not structured e-mails with Subject
and Body fields — median 14 words, p95 35 words. Everything in this project is
therefore described as *actionable email-content classification*; no component
claims to consume a Subject + Body structure.

### Inspection

`src/preprocessing/prepare_data.py` checks, before anything is trained: parse
failures, unknown labels, missing values, encoding damage, duplicates,
duplicates carrying conflicting labels, class distribution and text-length
distribution. Machine-readable output: `results/data_quality_report.json`.

### Removals

Every removal is logged with its reason in `results/removed_examples.csv`.

| Reason | Count |
|---|---|
| Malformed rows | 0 |
| Empty / meaningless after cleaning | 0 |
| Duplicate text with conflicting labels | 0 |
| Exact duplicates (same text and label) | 4 |
| **Total removed** | **4** |

Only exact duplicates were dropped, and only because a duplicate straddling the
train/test boundary would put training text into the test set. Rows whose text
appears under both labels would have been dropped as genuinely ambiguous rather
than relabelled — this dataset contains none. Promotional, newsletter and casual
e-mails were kept: they are valid negative examples, not noise.

Final corpus: **3,653 rows** (1,718 `Yes`, 1,935 `No`).

### Preprocessing policy

Repair the text; never destroy the signal (`src/preprocessing/clean.py`):

- Unicode NFKC normalisation, mojibake repair, HTML unescaping, HTML tag removal
- whitespace collapsing, control-character removal
- placeholder tokens for URLs (`$LINK`, the convention already present in the
  dataset), e-mail addresses (`$EMAIL`) and long digit strings (`$NUM`)

Explicitly **not** removed: stopwords, modal verbs, auxiliary verbs, action
verbs, temporal expressions, politeness markers. "Please", "confirm", "submit",
"review", "by tomorrow", "ASAP" and the question mark are precisely what
distinguishes an actionable e-mail from an informational one. A log-odds view of
the vocabulary (`notebooks/exploration.ipynb`, §4) confirms this: the actionable
side of the vocabulary is dominated by requests, second-person address and
deadlines.

The same conceptual policy feeds all three models; only the tokenisation differs,
as each architecture requires (word regex for the LSTM, WordPiece for BERT,
scikit-learn's analyser for TF-IDF).

## 3. Splitting

One stratified 70/15/15 split, seed 42, written once to `data/splits/` and
reused by every model:

| Split | Rows | `No` | `Yes` |
|---|---|---|---|
| train | 2,557 | 1,355 | 1,202 |
| val | 548 | 290 | 258 |
| test | 548 | 290 | 258 |

The three models are trained on the same 2,557 examples and scored on the same
548 test examples, so differences in the results are attributable to the
approach rather than to the data.

### Leakage control

- TF-IDF is fitted on the training split only — vocabulary and idf weights never
  see validation or test text.
- The LSTM vocabulary is built from training text only; unseen words map to `<unk>`.
- BERT is fine-tuned on training data only.
- Model selection uses training/validation data only: 5-fold cross-validated grid
  search **inside the training split** for Model A, validation-F1 early stopping
  and checkpointing for Models B and C.
- The test split is loaded once, at the end of each script, for scoring only.
- `tests/test_pipeline.py` asserts that the splits are disjoint in both ids and
  text, that they are stratified, and that `label_id` still matches the original
  label.

## 4. Models

### Model A — TF-IDF + Logistic Regression

TF-IDF is the feature extractor; Logistic Regression is the model. Word 1–2
grams, `sublinear_tf`, no stop-word list, `class_weight="balanced"`,
`liblinear` solver. Hyper-parameters (`ngram_range`, `min_df`, `C`) are chosen by
5-fold cross-validated grid search on the training split, scoring F1 on the
actionable class. Config: `configs/tfidf.json`.

### Model B — LSTM

Learned embedding (128d, trained jointly with the recurrent layer — no
pretrained vectors), 1-layer bidirectional LSTM with 128 hidden units, dropout
0.3, dense classification layer on the concatenated final hidden states.
Sequences are packed so padding never enters the recurrence. `max_len` 64 covers
above the 99th percentile of the corpus. Adam, lr 1e-3, batch 32, gradient
clipping 5.0, up to 20 epochs with early stopping (patience 4) and checkpointing
on validation F1. Config: `configs/lstm.json`. No separate vanilla RNN is
implemented; the LSTM is the recurrent approach.

### Model C — BERT

Standard `bert-base-uncased` with a `AutoModelForSequenceClassification` head.
**No layers are frozen and no architectural modifications are made** — the full
encoder is fine-tuned, so the result is comparable to the standard BERT
experiment. Small-dataset practice: AdamW, lr 2e-5, weight decay 0.01, 10%
linear warmup then linear decay, batch 16, `max_len` 64, gradient clipping 1.0,
up to 4 epochs with early stopping (patience 2) and checkpointing on validation
F1, fixed seed. Config: `configs/bert.json`.

A `--freeze-encoder` flag exists for anyone who wants to run the frozen-encoder
variant deliberately. It is **off by default**, it prints a warning that the run
is not the standard fine-tuning experiment, and the setting is recorded in the
saved config. Nothing is ever frozen silently.

If GPU memory is short, the documented remedy is to reduce `--batch-size` and
raise `--gradient-accumulation-steps` to keep the effective batch size, or to
reduce `--max-len` — not to alter the architecture.

## 5. Fair comparison

Same dataset, same labels, same partition, same test examples, same metric code
(`src/evaluation/metrics.py`) for all three models. Hyper-parameters differ
because the architectures differ, and each model was given a reasonable,
documented configuration rather than one model being tuned heavily and another
left arbitrary. Every configuration actually used is saved to
`models/<model>/config.json` and embedded in `results/<model>/metrics.json`.

## 6. Evaluation

For every model, on both the validation and test splits: accuracy, precision,
recall, F1, macro/weighted F1, ROC-AUC, the full confusion matrix and the
TP/TN/FP/FN counts. Accuracy alone is never reported.

The positive class is Actionable, so a **false negative is an actionable e-mail
predicted non-actionable** — the user misses something that needed their
attention. This is the error class that matters most for this application, and
it is why FN is reported explicitly in every table. The demo exposes
`--threshold` for exactly this trade-off: lowering it accepts more false
positives in exchange for fewer missed actionable e-mails.

## 7. Reproducibility

Fixed seeds (Python, NumPy, PyTorch, cuDNN deterministic) via
`src/common/seeding.py`; splits saved to disk and reused; configs, training
histories, metrics, confusion matrices and test predictions written to
`results/` as CSV/JSON; dependencies pinned by lower bound in
`requirements.txt`. No Kaggle path is hard-coded anywhere in `src/` — dataset,
results and model directories, batch size, epochs, learning rate, max length and
seed are all command-line arguments, which is what makes
`notebooks/training/kaggle_train_all.ipynb` a thin wrapper over the same code.
