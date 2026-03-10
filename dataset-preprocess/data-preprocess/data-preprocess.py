import pandas as pd
import os
import re

BASE_DIR = os.path.dirname(__file__)

TRAIN_INPUT = os.path.join(BASE_DIR, "train_cleaned.csv")
VALID_INPUT = os.path.join(BASE_DIR, "valid_cleaned.csv")
TEST_INPUT  = os.path.join(BASE_DIR, "test_cleaned.csv")

TRAIN_OUTPUT = os.path.join(BASE_DIR, "train_ready.csv")
VALID_OUTPUT = os.path.join(BASE_DIR, "valid_ready.csv")
TEST_OUTPUT  = os.path.join(BASE_DIR, "test_ready.csv")


def load_csv_with_fallback(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read file with tested encodings: {path}")


def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # lowercase normalization
    text = text.lower()

    # replace URL with token
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)

    # replace email with token
    text = re.sub(r"\S+@\S+", " email ", text)

    # replace phone numbers with token
    text = re.sub(r"\b\d{8,15}\b", " phone ", text)

    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # remove extra underscores
    text = re.sub(r"_+", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_file(input_path, output_path, file_name="dataset"):
    df = load_csv_with_fallback(input_path)

    required_cols = ["text", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{file_name}: Missing columns {missing_cols}. Dataset must contain: text,label"
        )

    # Keep only the required columns
    df = df[["text", "label"]].copy()

    # Ensure label is numeric integer
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2])]

    # Apply text preprocessing
    df["text"] = df["text"].apply(preprocess_text)

    # Remove empty rows after preprocessing
    df = df[
        df["text"].notna() &
        (df["text"] != "") &
        (df["text"].str.lower() != "nan") &
        (df["text"].str.lower() != "none")
    ].reset_index(drop=True)

    # Save processed file
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{file_name} preprocessing completed.")
    print(f"Saved to: {output_path}")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
    print("Sample rows:")
    print(df.head())


def main():
    process_file(TRAIN_INPUT, TRAIN_OUTPUT, "train")
    process_file(VALID_INPUT, VALID_OUTPUT, "valid")
    process_file(TEST_INPUT, TEST_OUTPUT, "test")

    print("\nAll preprocessing steps completed successfully.")


if __name__ == "__main__":
    main()