# NLP-mid
Midterm for NLP
Cam Volt
Phát hiện lừa đảo, spam, rao vặt trong comment

# Dataset 
https://www.kaggle.com/datasets/kazanova/sentiment140                                for normal comment 
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download  for scam comment
youtube+spam+collection                                                              for spam comment

# Train models
Train in `dataset-preprocess/data-preprocess`:

```bash
python dataset-preprocess/data-preprocess/train_linear_svm.py
python dataset-preprocess/data-preprocess/train_logistic_regression.py
```

Input files:
- `dataset-preprocess/data-preprocess/train_ready.csv`
- `dataset-preprocess/data-preprocess/valid_ready.csv`
- `dataset-preprocess/data-preprocess/test_ready.csv`

Output files are saved in:
- `dataset-preprocess/data-preprocess/artifacts/linear_svm`
- `dataset-preprocess/data-preprocess/artifacts/logistic_regression`
