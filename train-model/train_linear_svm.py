import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

LABEL_MAP = {0: "normal", 1: "spam", 2: "scam"}
VALID_LABELS = sorted(LABEL_MAP.keys())
# BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(r"D:\NLP-mid\dataset-preprocess\data-preprocess")
MODEL_FILENAME = "linear_svm_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Linear SVM")
    parser.add_argument("--train-path", type=Path, default=BASE_DIR / "train_ready.csv")
    parser.add_argument("--valid-path", type=Path, default=BASE_DIR / "valid_ready.csv")
    parser.add_argument("--test-path", type=Path, default=BASE_DIR / "test_ready.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts" / "linear_svm")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    return parser.parse_args()


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file: {path}")


def load_dataset(path: Path, split_name: str):
    if not path.exists():
        raise FileNotFoundError(f"{split_name} not found: {path}")

    df = load_csv_with_fallback(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{split_name} must contain text,label columns")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)

    text_mask = (df["text"].str.strip() != "") & (~df["text"].str.lower().isin(["nan", "none"]))
    df = df[text_mask].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{split_name} is empty after validation")

    return df["text"], df["label"].to_numpy()


def save_confusion_matrix_plot(cm: np.ndarray, title: str, output_path: Path) -> None:
    labels = [LABEL_MAP[idx] for idx in VALID_LABELS]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, split_name: str, output_dir: Path):
    report = classification_report(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        target_names=[LABEL_MAP[idx] for idx in VALID_LABELS],
        output_dict=True,
        zero_division=0,
    )

    with (output_dir / f"{split_name}_classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{LABEL_MAP[idx]}" for idx in VALID_LABELS],
        columns=[f"pred_{LABEL_MAP[idx]}" for idx in VALID_LABELS],
    )
    cm_df.to_csv(output_dir / f"{split_name}_confusion_matrix.csv", encoding="utf-8-sig")
    save_confusion_matrix_plot(cm, f"{split_name.upper()} Confusion Matrix", output_dir / f"{split_name}_confusion_matrix.png")

    return {
        "accuracy": round(float(report["accuracy"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "macro_precision": round(float(report["macro avg"]["precision"]), 6),
        "macro_recall": round(float(report["macro avg"]["recall"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train_text, y_train = load_dataset(args.train_path, "train")
    x_valid_text, y_valid = load_dataset(args.valid_path, "valid")
    x_test_text, y_test = load_dataset(args.test_path, "test")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,         # keep at most x vocabulary features unigram/bigram
        ngram_range=(1, 2),                     # use both unigrams(1 word/time) and bigrams(2 word/time)
        min_df=args.min_df,                     # ignore words/phrases that appear in fewer than 2 documents
        lowercase=False,                        # not force lowercase, because we setup lowercase at the beginning
        sublinear_tf=True,                      # use logarithmic term frequency scaling    
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_valid = vectorizer.transform(x_valid_text)
    x_test = vectorizer.transform(x_test_text)

    best_c = None
    best_model = None
    best_valid_macro_f1 = -1.0

    for c in args.c_values:
        model = LinearSVC(C=c, random_state=42)
        model.fit(x_train, y_train)
        valid_pred = model.predict(x_valid)
        valid_macro_f1 = f1_score(y_valid, valid_pred, average="macro")
        if valid_macro_f1 > best_valid_macro_f1:
            best_valid_macro_f1 = valid_macro_f1
            best_c = c
            best_model = model

    if best_model is None:
        raise RuntimeError("Linear SVM training failed")

    valid_pred = best_model.predict(x_valid)
    test_pred = best_model.predict(x_test)
    valid_metrics = evaluate_split(y_valid, valid_pred, "valid", args.output_dir)
    test_metrics = evaluate_split(y_test, test_pred, "test", args.output_dir)

    model_path = args.output_dir / MODEL_FILENAME
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": best_model, "label_map": LABEL_MAP},
        model_path,
    )

    summary = {
        "model_name": "linear_svm",
        "best_c": best_c,
        "dataset_paths": {
            "train": str(args.train_path),
            "valid": str(args.valid_path),
            "test": str(args.test_path),
        },
        "vectorizer": {
            "max_features": args.max_features,
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "vocab_size": int(len(vectorizer.get_feature_names_out())),
        },
        "class_names": LABEL_MAP,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "model_file": str(model_path),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Linear SVM training completed.")
    print(f"Output dir: {args.output_dir}")
    print(f"Model file: {model_path.name}")
    print(
        f"Best C={best_c} | valid_macro_f1={valid_metrics['macro_f1']} | "
        f"test_macro_f1={test_metrics['macro_f1']}"
    )


if __name__ == "__main__":
    main()
