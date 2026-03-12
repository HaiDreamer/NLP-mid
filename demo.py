from pathlib import Path
import joblib

MODEL_PATH = Path(r"D:\NLP-mid\train-model\artifacts\linear_svm\linear_svm_model.joblib")

def load_model_bundle(model_path: Path):
    bundle = joblib.load(model_path)

    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    label_map = bundle["label_map"]

    return vectorizer, classifier, label_map

def predict_text(text: str):
    vectorizer, classifier, label_map = load_model_bundle(MODEL_PATH)

    # Convert raw text into TF-IDF features
    x = vectorizer.transform([text])

    # Predict numeric label
    pred_id = int(classifier.predict(x)[0])

    # Convert numeric label to class name
    pred_name = label_map[pred_id]

    return pred_id, pred_name

if __name__ == "__main__":
    while True:
        user_text = input("Enter text (or 'quit'): ").strip()
        if user_text.lower() == "quit":
            break

        pred_id, pred_name = predict_text(user_text)
        print(f"Predicted label: {pred_id} -> {pred_name}\n")