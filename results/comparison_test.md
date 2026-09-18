# MailSense -- model comparison (test set)

Positive class = **Actionable (Yes)**. FN = an actionable e-mail predicted non-actionable (the costly error for this application).

| model | accuracy | precision | recall | f1 | roc_auc | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Model A: TF-IDF + Logistic Regression | 0.8029 | 0.7799 | 0.8101 | 0.7947 | 0.8891 | 209 | 231 | 59 | 49 |
| Model B: LSTM | 0.8029 | 0.7737 | 0.8217 | 0.7970 | 0.8747 | 212 | 228 | 62 | 46 |
| Model C: BERT | 0.8522 | 0.8266 | 0.8682 | 0.8469 | 0.9343 | 224 | 243 | 47 | 34 |
