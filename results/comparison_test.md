# MailSense -- model comparison (test set)

Positive class = **Actionable (Yes)**. FN = an actionable e-mail predicted non-actionable (the costly error for this application).

| model | accuracy | precision | recall | f1 | roc_auc | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Model A: TF-IDF + Logistic Regression | 0.7774 | 0.7698 | 0.7519 | 0.7608 | 0.8607 | 194 | 232 | 58 | 64 |
| Model B: LSTM | 0.7482 | 0.7174 | 0.7674 | 0.7416 | 0.8654 | 198 | 212 | 78 | 60 |
| Model C: BERT | 0.8485 | 0.8205 | 0.8682 | 0.8437 | 0.9316 | 224 | 241 | 49 | 34 |
