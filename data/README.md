# Data

## Source file

`Ask0729-fixed.txt` — the provided Enron-based actionable e-mail dataset.

Format: one example per line, `label<TAB>text`, UTF-8.

| Property | Value |
|---|---|
| Rows | 3,657 |
| Actionable (`Yes`) | 1,719 |
| Non-Actionable (`No`) | 1,938 |
| Malformed rows | 0 |
| Missing values | 0 |

The examples are **e-mail sentences / snippets**, not structured e-mails with
separate Subject and Body fields. The project is therefore described honestly as
**actionable email-content/text classification**, and no part of the code
pretends to consume a Subject + Body structure.

## Labels

The original labels are authoritative and are kept verbatim in every processed
file. The numeric column is a deterministic mapping and nothing else:

| Original label | `label_id` | Meaning |
|---|---|---|
| `No` | 0 | Non-Actionable |
| `Yes` | 1 | Actionable |

No example is ever relabelled by hand.

## Generated files

`python -m src.preprocessing.prepare_data` produces:

```
data/processed/mailsense_clean.csv   cleaned corpus (id, label, label_id, text, raw_text, lengths)
data/splits/train.csv                2,557 rows
data/splits/val.csv                    548 rows
data/splits/test.csv                   548 rows
```

The split is stratified by label, seeded (`--seed 42`), written once, and reused
by all three models so the comparison is on identical data.

## Cleaning and removals

Cleaning repairs the text without destroying actionability cues: whitespace
normalisation, Unicode/mojibake repair, HTML unescaping and tag removal, and
placeholder tokens for URLs (`$LINK`, the convention already used in the
dataset), e-mail addresses (`$EMAIL`) and long digit strings (`$NUM`).

Stopwords, modal and auxiliary verbs, action verbs, temporal expressions and
politeness markers ("please", "ASAP", "deadline", …) are **kept** — they are the
signal for this task. Promotional / newsletter / casual e-mails are kept as
valid negative examples.

Removals from the current run (4 rows in total, all logged in
`results/removed_examples.csv` with the reason):

| Reason | Count |
|---|---|
| Malformed rows | 0 |
| Empty / meaningless after cleaning | 0 |
| Duplicate text with conflicting labels | 0 |
| Exact duplicates (same text and label) | 4 |

Exact duplicates are dropped because a duplicate straddling the train/test
boundary would leak training text into the test set. Rows whose text appears
with both labels would be dropped as ambiguous rather than relabelled — the
current dataset contains none.

After cleaning: **3,653 rows** — 1,718 `Yes`, 1,935 `No`.

Text length (cleaned, in words): median 14, mean 16.6, p95 35, p99 51, max 222.
This is why `max_len = 64` is used for both the LSTM and BERT.

## Full-dataset statistics

Machine-readable versions of everything above live in
`results/data_quality_report.json`.
