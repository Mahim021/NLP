# MailSense -- model comparison (val set)

Positive class = **Actionable (Yes)**. FN = an actionable e-mail predicted non-actionable (the costly error for this application).

| model | accuracy | precision | recall | f1 | roc_auc | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Model A: TF-IDF + Logistic Regression | 0.8029 | 0.7953 | 0.7829 | 0.7891 | 0.8760 | 202 | 238 | 52 | 56 |
| Model B: LSTM | 0.7920 | 0.7535 | 0.8295 | 0.7897 | 0.8702 | 214 | 220 | 70 | 44 |
| Model C: BERT | 0.8577 | 0.8261 | 0.8837 | 0.8539 | 0.9286 | 228 | 242 | 48 | 30 |
