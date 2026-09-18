
You are the implementation assistant for this project. Do not redesign the project. Your job is to implement, test, document, and improve the specified system while preserving the project's research question and experimental integrity. When you believe a change is necessary, explain it before making it.



# MailSense — Project Specification

## 1. Project Identity

Project title:

MailSense: Actionable Email Detection Using NLP

Authors:

Md Ariful Alam Mahim (Roll: 2107023)
Md Jubair Husain (Roll: 2107029)

This document is the authoritative specification for the project.

Claude must follow this specification unless explicitly instructed by the
developers/users to change it.

Do not silently change the problem definition, dataset, labels, models,
evaluation protocol, or project scope.

---

# 2. Problem Definition

MailSense is a binary NLP classification system that determines whether
an email text requires action/attention from the recipient.

The two classes are:

Yes -> Actionable
No  -> Non-Actionable

The original dataset labels must be preserved.

Do NOT invent additional intent classes.

Do NOT convert the problem into:
- meeting detection
- submission detection
- teacher/student intent classification
- email reply generation
- sentiment analysis
- spam detection
- email summarization

The central question is:

"Does this email content require the recipient's action or attention?"

---

# 3. Dataset

The project uses the provided Enron-based actionable email dataset.

The dataset contains 3,657 examples:

Actionable (Yes): 1,719
Non-Actionable (No): 1,938

The dataset consists primarily of email sentences/snippets rather than
structured full emails with separate Subject and Body fields.

Therefore, the project must honestly be described as:

"actionable email-content/text classification"

Do NOT claim that the system has been trained on structured Subject + Body
emails when the dataset does not contain that structure.

Do NOT replace the dataset with another dataset unless explicitly instructed.

Do NOT generate synthetic training examples unless explicitly instructed.

Do NOT manually relabel examples based on your own interpretation.

---

# 4. Dataset Integrity

The original labels are authoritative:

Yes = Actionable
No = Non-Actionable

Keep the original label in the processed data whenever practical.

Any derived label such as:

Actionable = 1
Non-Actionable = 0

must be a deterministic mapping from the original label.

Never silently alter labels.

Before training, inspect:
- missing values
- malformed rows
- encoding problems
- duplicates
- suspicious/corrupted entries
- class distribution
- text length distribution

Document any removed examples and the reason for removal.

Do not aggressively remove legitimate promotional/newsletter/casual emails
merely because they are non-actionable. They are valid negative examples.

---

# 5. Preprocessing

Use appropriate NLP preprocessing, but do not destroy information that may
be useful for actionability detection.

Reasonable operations may include:
- whitespace normalization
- handling malformed encoding
- cleaning obvious HTML artifacts
- normalizing URLs where appropriate
- removing clearly corrupted/meaningless fragments

Do NOT blindly remove:
- stopwords
- modal verbs
- auxiliary verbs
- action verbs
- temporal expressions
- words such as "please", "submit", "confirm", "review", "send", "deadline",
  "tomorrow", "ASAP", etc.

These may be important for detecting actionability.

Use the same conceptual preprocessing policy consistently across models,
while allowing model-specific tokenization required by each architecture.

---

# 6. Models

The project will compare exactly three primary approaches.

## Model A: TF-IDF + Logistic Regression

Pipeline:

Text
-> preprocessing
-> TF-IDF
-> Logistic Regression
-> binary prediction

This is the traditional machine-learning baseline.

Do not call TF-IDF itself a model.

---

## Model B: LSTM

Pipeline:

Text
-> preprocessing
-> tokenizer
-> token IDs
-> embedding
-> LSTM
-> dense classification layer
-> binary prediction

The embedding may be learned jointly with the LSTM from the training data.

Do not implement a separate RNN model.

LSTM is sufficient as the recurrent neural-network approach.

---

## Model C: BERT

Pipeline:

Text
-> BERT tokenizer
-> pretrained BERT
-> classification head
-> binary prediction

Use an established pretrained BERT implementation rather than implementing
the Transformer architecture from scratch.

The purpose is to evaluate a pretrained Transformer-based contextual model.

BERT must not be replaced by another architecture without explicit approval.

---

# 7. BERT Rules

Do not make arbitrary architectural modifications to BERT.

Do not remove layers/features merely to make the model easier.

Start with a standard pretrained BERT sequence-classification setup.

Use reasonable practices for a small dataset, including:
- validation set
- early stopping where appropriate
- checkpointing
- appropriate learning rate
- appropriate batch size
- maximum sequence length appropriate for the dataset
- reproducible random seeds

If computational constraints arise, prefer documented changes such as:
- reducing batch size
- reducing maximum sequence length
- using gradient accumulation if necessary
- using a smaller pretrained BERT variant if justified

Freezing BERT layers is an experimental choice, not an automatic requirement.

If freezing is proposed, explain:
1. what is being frozen
2. why
3. what effect it may have
4. whether the result remains comparable to the standard BERT experiment

Do not silently freeze layers.

---

# 8. Data Splitting

Use a stratified train/validation/test split.

The exact split should be defined once and reused across models.

All three models must be evaluated on the SAME test set.

Do not allow any test data to influence training, vocabulary fitting,
TF-IDF fitting, embedding training, model selection, or hyperparameter tuning.

Fit TF-IDF only on the training data.

Build learned vocabulary/tokenization resources without leaking test
information.

BERT must also never be trained or selected using the test set.

---

# 9. Fair Comparison

The goal is to compare approaches, not to artificially make one model win.

Use the same:
- dataset
- labels
- train/test partition
- evaluation metrics
- test examples

for all models.

Hyperparameters may differ because the architectures are different.

Do not tune one model extensively while leaving another at an arbitrary
configuration and then present the comparison as definitive.

Record important hyperparameters and training settings.

---

# 10. Evaluation

For every model report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Also report the number of:
- True Positives
- True Negatives
- False Positives
- False Negatives

False Negatives are particularly important for this application because
an actionable email incorrectly classified as non-actionable could cause
the user to miss something requiring attention.

Do not optimize or report only accuracy.

---

# 11. Experimental Reproducibility

The project must be reproducible.

Use:
- fixed random seeds where practical
- saved train/validation/test splits
- saved model configurations
- saved training histories
- saved evaluation results
- clear requirements/dependency files

Never hard-code Kaggle-specific paths into the core project.

The project must work with configurable dataset paths.

---

# 12. Git

Maintain a clean Git repository.

Recommended structure:

mail-sense/
|
|-- data/
|   |-- README.md
|   |-- dataset file
|
|-- notebooks/
|   |-- exploration.ipynb
|   |-- training/
|
|-- src/
|   |-- preprocessing/
|   |-- tfidf/
|   |-- lstm/
|   |-- bert/
|   |-- evaluation/
|
|-- configs/
|
|-- results/
|
|-- models/
|
|-- reports/
|
|-- requirements.txt
|-- README.md
|-- PROJECT_SPEC.md
|-- .gitignore

Do not commit large model checkpoints or unnecessary generated files
unless explicitly required.

Use Git-friendly outputs such as:
- CSV result tables
- configuration files
- training logs
- evaluation summaries

---

# 13. Kaggle

The actual heavy training will be performed on Kaggle GPU when necessary.

The code developed locally must be transferable to Kaggle.

Do not create code that depends unnecessarily on local hardware.

The training scripts should accept configurable:
- dataset path
- output path
- batch size
- epochs
- learning rate
- model configuration
- random seed

Training must save:
- best checkpoint
- final checkpoint when useful
- training history
- validation metrics
- test metrics
- confusion matrix data

---

# 14. Engineering Philosophy

Keep the implementation simple and understandable.

Do not add:
- unnecessary GUI frameworks
- unnecessary APIs
- cloud databases
- email-server integration
- email sending
- automatic reply generation
- complicated deployment
- unnecessary model architectures

unless explicitly requested.

The final demonstration only needs to show:

Input email text
-> MailSense
-> Actionable / Non-Actionable

with an optional confidence/probability.

---

# 15. Do Not Overfit to Coursework Topics

Do not implement a separate experiment for every NLP technique learned
in the laboratory.

Only include techniques that contribute to the project.

Current primary approaches are:

1. TF-IDF + Logistic Regression
2. LSTM
3. BERT

Do not add BoW, vanilla RNN, GRU, CNN, Word2Vec training, GloVe training,
FastText training, or other models unless explicitly requested.

---

# 16. Development Rules

Before making a major architectural decision:

1. Check PROJECT_SPEC.md.
2. Explain why the change is needed.
3. Prefer the simplest valid solution.
4. Do not silently change project scope.
5. Keep experimental changes reproducible.
6. Record important decisions in documentation.

If there are multiple reasonable choices, present the alternatives and
recommend one based on:
- simplicity
- correctness
- reproducibility
- suitability for the dataset
- computational requirements

Do not introduce complexity merely because it is technically possible.

---

# 17. Expected Final Research Question

The project should ultimately answer:

"How do a traditional TF-IDF-based classifier, an LSTM-based neural model,
and a pretrained Transformer-based BERT model perform for actionable
email-content classification on our dataset?"

The final report should compare the actual experimental results rather
than assuming beforehand which model will perform best.