# NLP Midterm Project
## Detecting Normal, Spam, and Scam in Comments

This project is a multi-class text classification system for detecting:

- **0 — Normal**: clean / non-harmful comments
- **1 — Spam**: advertisements, promotional content, rao vặt
- **2 — Scam**: fraudulent, phishing, or deceptive content

The goal is to build machine learning models that can classify user-generated text into these three categories.

---

## Dataset

This project combines data from multiple public datasets:

### 1. Normal class
- **Sentiment140**
- Source: https://www.kaggle.com/datasets/kazanova/sentiment140

Used as the **normal / clean** text class.

### 2. Spam class
- **SMS Spam Collection Dataset**
- Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

Used for **spam / advertising / promotional** content.

### 3. Scam class
- **Phishing Emails Dataset**
- Source: https://www.kaggle.com/datasets/subhajournal/phishingemails
- Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

Used for **scam / phishing / fraudulent** content.

---

## Labels

The dataset is converted into a 3-class classification problem:

- `0` → normal, clean
- `1` → spam, rao vặt
- `2` → scam, lừa đảo

---

## Data Files

Training and evaluation use the following processed files:

- `dataset-preprocess/data-preprocess/train_ready.csv`
- `dataset-preprocess/data-preprocess/valid_ready.csv`
- `dataset-preprocess/data-preprocess/test_ready.csv`

---

## Train Models

Two traditional machine learning models are provided:
```bash
python dataset-preprocess/data-preprocess/train_linear_svm.py
python dataset-preprocess/data-preprocess/train_logistic_regression.py
```

# Use model/demo
Run: app_demo.py for app use or demo.py for use model directly
Note: only available for svm model

# Limitation
Limited dataset
English