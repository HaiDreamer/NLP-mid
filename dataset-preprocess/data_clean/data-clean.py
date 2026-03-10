import pandas as pd
import os
import html
import re

BASE_DIR = os.path.dirname(__file__)

# Change input/output file names here
INPUT_PATH = os.path.join(BASE_DIR, "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "test_cleaned.csv")
CONFLICT_PATH = os.path.join(BASE_DIR, "test_conflicts.csv")


def load_csv_with_fallback(path):
    # Try multiple encodings to safely read the file
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read file with tested encodings: {path}")


def clean_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    # 1. Load dataset
    df = load_csv_with_fallback(INPUT_PATH)

    # 2. Check required columns
    required_cols = ["text", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns {missing_cols}. Dataset must contain: text,label"
        )

    # 3. Keep only necessary columns
    df = df[["text", "label"]].copy()

    # 4. Clean label column
    # Convert label to numeric (float -> int, string -> number)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Remove rows with invalid labels
    df = df.dropna(subset=["label"])

    # Convert labels to integer
    df["label"] = df["label"].astype(int)

    # Keep only valid labels
    df = df[df["label"].isin([0, 1, 2])]

    # 5. Clean text content
    df["text"] = df["text"].apply(clean_text)

    # 6. Remove empty or invalid text rows
    df = df[
        df["text"].notna() &
        (df["text"] != "") &
        (df["text"].str.lower() != "nan") &
        (df["text"].str.lower() != "none")
    ]

    # 7. Remove exact duplicates (same text and label)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # 8. Detect conflicts: same text but different labels
    label_count_per_text = df.groupby("text")["label"].nunique()
    conflict_texts = label_count_per_text[label_count_per_text > 1].index.tolist()

    if conflict_texts:
        conflict_df = df[df["text"].isin(conflict_texts)].copy()
        conflict_df.to_csv(CONFLICT_PATH, index=False, encoding="utf-8-sig")
        print(f"Conflict file saved: {CONFLICT_PATH}")
        print(f"Number of texts assigned multiple labels: {len(conflict_texts)}")

        # Remove conflicting samples from the clean dataset
        df = df[~df["text"].isin(conflict_texts)].reset_index(drop=True)
    else:
        print("No conflicting labels found.")

    # 9. Remove remaining duplicates based only on text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # 10. Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 11. Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nCleaning completed!")
    print("Saved cleaned dataset to:", OUTPUT_PATH)

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nData types:")
    print(df.dtypes)

    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()