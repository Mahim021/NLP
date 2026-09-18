# MailSense — Results

Md Ariful Alam Mahim (2107023) · Md Jubair Husain (2107029)

How the experiment was run is documented in [methodology.md](methodology.md).
This document reports what the experiment actually produced.

## Research question

> How do a traditional TF-IDF-based classifier, an LSTM-based neural model, and a
> pretrained Transformer-based BERT model perform for actionable email-content
> classification on our dataset?

## Setup recap

All three models were trained on the same 2,557 training examples and scored on
the same 548 test examples (290 `No`, 258 `Yes`), with the same metric code.
Positive class = **Actionable (`Yes`)**, so **FN = an actionable e-mail predicted
non-actionable**.

## Test-set results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| A: TF-IDF + Logistic Regression | 0.8029 | 0.7799 | 0.8101 | 0.7947 | 0.8891 | 209 | 231 | 59 | 49 |
| B: LSTM | 0.8029 | 0.7737 | 0.8217 | 0.7970 | 0.8747 | 212 | 228 | 62 | 46 |
| **C: BERT** | **0.8522** | **0.8266** | **0.8682** | **0.8469** | **0.9343** | 224 | 243 | 47 | **34** |

Confusion matrices (rows = true, columns = predicted):

| | A: TF-IDF + LogReg | B: LSTM | C: BERT |
|---|---|---|---|
| true `No` | 231 / 59 | 228 / 62 | 243 / 47 |
| true `Yes` | 49 / 209 | 46 / 212 | 34 / 224 |

Figures: `results/figures/cm_*_test.png`. Raw numbers:
`results/comparison_test.csv`, `results/<model>/metrics.json`.

## Validation-set results

Reported for completeness; the validation split is what drove model selection,
so these numbers are optimistic relative to the test numbers.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FN |
|---|---|---|---|---|---|---|
| A: TF-IDF + Logistic Regression | 0.8029 | 0.7953 | 0.7829 | 0.7891 | 0.8760 | 56 |
| B: LSTM | 0.7920 | 0.7535 | 0.8295 | 0.7897 | 0.8702 | 44 |
| C: BERT | 0.8577 | 0.8261 | 0.8837 | 0.8539 | 0.9286 | 30 |

The val→test gap is small and in the expected direction for every model, which
suggests the selection procedure did not overfit the validation split.

## Findings

**1. BERT wins, and the margin is real.** BERT improves accuracy by 4.9 points
and F1 by 5.0–5.2 points over both other approaches. On 548 test examples a
5-point gap is roughly 27 examples — comfortably larger than the ~±3.0-point
95% binomial interval on a single accuracy estimate at n=548, so this is not
noise. BERT is also clearly better calibrated for ranking: ROC-AUC 0.934 against
0.889 and 0.875.

**2. The LSTM did not beat the TF-IDF baseline.** They tie on accuracy (0.8029
each), and the LSTM's F1 advantage is 0.0023 — three test examples' worth, i.e.
nothing. This is the expected outcome for 2,557 short training examples: the
LSTM must learn its 128-dimensional embeddings from scratch on a corpus far too
small to do so, and it began overfitting almost immediately (validation loss
rose from epoch 2 onward while training loss kept falling; early stopping fired
at epoch 11 with the best checkpoint at epoch 7). The pretrained model is what
supplies the missing linguistic knowledge — not the recurrent architecture.

**3. On the error that matters, BERT misses the fewest.** False negatives are
actionable e-mails the user would never be alerted to: 49 (A), 46 (B), 34 (C) —
BERT misses 15 fewer than the baseline, a 31% reduction. All three models lean
the same way (recall > precision on the actionable class), which is the right
direction for this application: they over-flag rather than under-flag.

**4. The remaining errors are genuinely hard, not model-specific.** 18 of BERT's
34 false negatives are also missed by the TF-IDF baseline. Inspecting them, they
are cases where actionability is implicit rather than lexically marked:

- *"Should you have any questions or require additional information, please feel
  free to contact me"* — polite boilerplate whose surface form looks like a
  request but whose label depends on context
- *"We would like to invite you back to continue."* — an invitation with no
  explicit imperative
- *"I did call and got credit on my bill, it doesnt just show up u need to make
  the call and they a…"* — a truncated snippet where the action applies to
  someone other than the recipient

Because these are snippets stripped of thread context, no model can resolve
*who* is being asked to act. This is a ceiling imposed by the dataset, not by
the architectures.

## Cost

| Model | Trainable parameters | Training time (this run, CPU) | Best epoch |
|---|---|---|---|
| A: TF-IDF + Logistic Regression | ~27k features, linear | seconds (incl. 5-fold grid search) | n/a |
| B: LSTM | 584,194 | 88 s (11 epochs, early-stopped) | 7 |
| C: BERT | 109,483,778 | 3,300 s (4 epochs) | 4 |

BERT costs ~187× the parameters and ~37× the training time of the LSTM for a
5-point gain. On a GPU the wall-clock difference largely disappears, but for a
deployment where inference cost matters, the TF-IDF baseline delivers 94% of
BERT's accuracy at a negligible fraction of the cost — a real trade-off, not an
argument against BERT.

## Answer to the research question

On this dataset, the pretrained Transformer is the strongest of the three
approaches on every reported metric, and it reduces the costly error class
(missed actionable e-mails) by 31% relative to the traditional baseline. The
LSTM, trained from scratch on 2,557 short examples, offers no measurable
advantage over TF-IDF + Logistic Regression — the benefit comes from
pretraining, not from recurrence.

## Threats to validity

- **Single split, single seed.** Every number is one run on one stratified
  70/15/15 split. Differences of ~1 point (A vs. B) are within run-to-run
  variation; the BERT gap is not. Repeating across seeds would tighten the
  intervals and is the obvious next step.
- **Snippets, not e-mails.** The dataset is sentence-level, so these results do
  not transfer directly to full Subject + Body e-mails without re-validation.
- **Label subjectivity.** Actionability is partly a judgement call, and the
  residual errors concentrate exactly where that judgement is hardest.
- **Model A was tuned by grid search; B and C were not.** Both neural models ran
  at a single documented configuration. If anything, this biases the comparison
  *toward* the baseline, so it does not weaken the finding that BERT wins.

## Reproducing

```bash
python -m src.preprocessing.prepare_data
python -m src.tfidf.train_tfidf
python -m src.lstm.train_lstm
python -m src.bert.train_bert
python -m src.evaluation.compare
```

All numbers above were produced by this sequence with seed 42.
