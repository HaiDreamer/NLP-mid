import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(__file__)

TRAIN_PATH = os.path.join(BASE_DIR, "train_ready.csv")
VALID_PATH = os.path.join(BASE_DIR, "valid_ready.csv")
TEST_PATH = os.path.join(BASE_DIR, "test_ready.csv")


def load_dataset(path, name="dataset"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    required_cols = ["text", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{name}: Missing columns {missing_cols}. Required columns are: text, label"
        )

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)

    return df


def main():
    # Load datasets
    train_df = load_dataset(TRAIN_PATH, "train")
    valid_df = load_dataset(VALID_PATH, "valid")
    test_df = load_dataset(TEST_PATH, "test")

    # Split text and label
    X_train_text = train_df["text"]
    y_train = train_df["label"]

    X_valid_text = valid_df["text"]
    y_valid = valid_df["label"]

    X_test_text = test_df["text"]
    y_test = test_df["label"]

    # TF-IDF vectorizer
    # converting documents into a numeric feature matrix, then training a classifier on that matrix
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=False
    )

    # Fit on train only, avoid data leak
    X_train = vectorizer.fit_transform(X_train_text)

    # Transform valid and test
    X_valid = vectorizer.transform(X_valid_text)
    X_test = vectorizer.transform(X_test_text)

    # Print information
    print("TF-IDF vectorization completed.\n")

    print("Train TF-IDF shape:", X_train.shape)
    print("Valid TF-IDF shape:", X_valid.shape)
    print("Test TF-IDF shape :", X_test.shape)

    print("\nLabel shapes:")
    print("y_train:", y_train.shape)
    print("y_valid:", y_valid.shape)
    print("y_test :", y_test.shape)

    feature_names = vectorizer.get_feature_names_out()
    print("\nNumber of features:", len(feature_names))
    print("First 30 features:")
    print(feature_names[:30])

    # inspect first sample
    print("\nFirst training text:")
    print(X_train_text.iloc[0])

    print("\nTop non-zero TF-IDF values in first training sample:")
    first_vector = X_train[0]
    indices = first_vector.nonzero()[1]

    for idx in indices[:20]:
        print(f"{feature_names[idx]}: {first_vector[0, idx]:.4f}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test, vectorizer


if __name__ == "__main__":
    main()