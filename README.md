# MailSense — Actionable Email Detection Using NLP

**Md Ariful Alam Mahim (2107023) · Md Jubair Husain (2107029)**

MailSense answers one question about a piece of e-mail text:

> **Does this e-mail content require the recipient's action or attention?**

It is a binary classifier over the provided Enron-based dataset, with the
original labels preserved:

- `Yes` = **Actionable**
- `No` = **Non-Actionable**

Because the dataset consists of e-mail **sentences / snippets** rather than
structured Subject + Body e-mails, the task is described throughout as
**actionable email-content classification**.

---

## Research Question

> How do a traditional TF-IDF-based classifier, an LSTM-based neural model,
> and a pretrained Transformer-based BERT model perform for actionable
> email-content classification on our dataset?

Three approaches are compared using the same dataset, labels, data split,
and held-out test set.

| | Approach | Pipeline |
|---|---|---|
| **A** | TF-IDF + Logistic Regression | text → preprocessing → TF-IDF → Logistic Regression → Yes/No |
| **B** | LSTM | text → preprocessing → tokenizer → pretrained Word2Vec embedding → LSTM → dense → Yes/No |
| **C** | BERT | text → BERT tokenizer → pretrained BERT → classification head → Yes/No |

### Model A — TF-IDF + Logistic Regression

TF-IDF is used as the text feature representation for a traditional machine
learning classifier. Logistic Regression performs the final binary
classification.

### Model B — LSTM

The LSTM receives token IDs that are mapped to pretrained Word2Vec word
embeddings. The embeddings are then processed by a bidirectional LSTM followed
by a fully connected classification layer.

The Word2Vec model is pretrained externally rather than trained from scratch
on the relatively small project dataset.

### Model C — BERT

A pretrained BERT model is fine-tuned for binary classification. The standard
pretrained BERT architecture is used together with a classification head.

---

## Dataset

The finalized dataset contains:

- **3,657 total examples**
- **1,719 Actionable (`Yes`)**
- **1,938 Non-Actionable (`No`)**

The original `Yes` / `No` labels are preserved.

The dataset consists primarily of individual e-mail sentences or snippets,
so the project evaluates **actionability of e-mail content**, rather than
claiming to model complete structured e-mails with separate Subject and Body
fields.

---

## Data Preparation

The preprocessing pipeline performs:

1. Unicode normalization
2. HTML decoding/removal
3. URL replacement with `$LINK`
4. E-mail address replacement with `$EMAIL`
5. Number replacement with `$NUM`
6. Whitespace normalization
7. Removal of empty or meaningless examples
8. Removal of duplicate examples after cleaning
9. Stratified train/validation/test splitting

The same finalized split is used by all three models.

### Dataset Split

- **70%** training
- **15%** validation
- **15%** test

The test set is held out during model training and model selection.

---

## Evaluation

All models are evaluated using the same held-out test set.

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

The positive class is:

**`Yes` = Actionable**

Therefore:

> **False Negative = an actionable e-mail predicted as non-actionable.**

This error is important for the application because it could cause the user
to miss an e-mail requiring attention.

---

## Results

### Final results

Final results will be added after the controlled Kaggle experiments.

The final comparison will be produced only after all three models are trained
using the finalized code and evaluated on the same held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: TF-IDF + Logistic Regression | — | — | — | — | — | — | — | — | — |
| B: LSTM | — | — | — | — | — | — | — | — | — |
| C: BERT | — | — | — | — | — | — | — | — | — |

Results from local development or smoke tests are not considered final
experimental results.

The final report will present the measured results without assuming in
advance which model will perform best.

---

## Repository Layout

```text
mail-sense/
│
├── data/
│   ├── README.md
│   ├── dataset
│   └── splits/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── training/
│       └── kaggle_train_all.ipynb
│
├── src/
│   ├── preprocessing/
│   │   └── prepare_data.py
│   │
│   ├── tfidf/
│   │   └── train_tfidf.py
│   │
│   ├── lstm/
│   │   ├── model.py
│   │   ├── train_lstm.py
│   │   └── vocab.py
│   │
│   ├── bert/
│   │   └── train_bert.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── compare.py
│   │
│   ├── common/
│   │   ├── seeding.py
│   │   ├── labels.py
│   │   ├── data.py
│   │   └── paths.py
│   │
│   └── predict.py
│
├── configs/
│   ├── tfidf.json
│   ├── lstm.json
│   └── bert.json
│
├── results/
│   ├── metrics
│   ├── confusion matrices
│   ├── training histories
│   └── comparison tables
│
├── models/
│   └── checkpoints
│
├── reports/
│   ├── methodology.md
│   └── results.md
│
├── requirements.txt
├── README.md
└── .gitignore