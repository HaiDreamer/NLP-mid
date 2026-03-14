from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

'''
run: streamlit run app_demo.py

confidence score algo:
    confidence score proportional to the signed distance to the hyperplane, bigger value is better
        margin near 0 → uncertain / near boundary
    Multiclass (we have 3 classes) case
        confidence_gap = top1_score - top2_score        => big gap means more confidence ranking
    Display
        Predicted label
        Top margin
        Margin gap vs 2nd class

'''

MODEL_PATH = Path(r"D:\NLP-mid\train-model\artifacts\linear_svm\linear_svm_model.joblib")

st.set_page_config(
    page_title="Text Classification Demo",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
.result-card {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: #f5f7fb;
    border: 1px solid #dbe2ea;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
.small-muted {
    color: #6b7280;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle(model_path: Path):
    bundle = joblib.load(model_path)

    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    label_map = bundle["label_map"]

    return vectorizer, classifier, label_map


def get_ranked_scores(classifier, x, label_map):
    """
    Return a DataFrame of classes ranked by decision margin.
    For SVM, these are margins, not probabilities.
    """
    if not hasattr(classifier, "decision_function"):
        return None

    raw_scores = classifier.decision_function(x)
    raw_scores = np.asarray(raw_scores)

    class_ids = np.asarray(classifier.classes_)

    # Multiclass case: raw_scores shape is usually (1, n_classes)
    if raw_scores.ndim == 2:
        scores = raw_scores[0]

    # Binary case: raw_scores may be shape (1,)
    elif raw_scores.ndim == 1 and len(class_ids) == 2 and raw_scores.size == 1:
        margin = float(raw_scores[0])
        scores = np.array([-margin, margin])

    else:
        return None

    rows = []
    for class_id, score in zip(class_ids, scores):
        class_id = int(class_id)
        rows.append({
            "label_id": class_id,
            "label_name": label_map[class_id],
            "margin": float(score),
        })

    df = pd.DataFrame(rows).sort_values("margin", ascending=False).reset_index(drop=True)
    return df


def predict_text(text: str, vectorizer, classifier, label_map):
    x = vectorizer.transform([text])

    pred_id = int(classifier.predict(x)[0])
    pred_name = label_map[pred_id]

    ranked_df = get_ranked_scores(classifier, x, label_map)

    return pred_id, pred_name, ranked_df


# ---------- UI ----------
st.title("🧠 Text Classification Demo")
st.caption("Enter text below and get the predicted class from trained Linear SVM model.")

# Safer startup error display
if not MODEL_PATH.exists():
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

vectorizer, classifier, label_map = load_model_bundle(MODEL_PATH)

sample_texts = {
    "Custom": "",
    "Example 1": "This movie was amazing. The acting and story were excellent.",
    "Example 2": "I am very disappointed with the service. It was slow and rude.",
    "Example 3": "The package arrived on time and the quality is acceptable.",
}

selected_example = st.selectbox("Choose an example", list(sample_texts.keys()))
default_text = sample_texts[selected_example]

user_text = st.text_area(
    "Input text",
    value=default_text,
    height=180,
    placeholder="Type or paste text here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("Predict", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    clean_text = user_text.strip()

    if not clean_text:
        st.warning("Please enter some text first.")
    else:
        pred_id, pred_name, ranked_df = predict_text(
            clean_text, vectorizer, classifier, label_map
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-muted">Prediction</div>
                <h2 style="margin: 0.2rem 0 0.5rem 0;">{pred_name}</h2>
                <div class="small-muted">Label ID: {pred_id}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if ranked_df is not None:
            st.subheader("Class ranking")
            st.caption("These are SVM decision margins, not calibrated probabilities.")
            st.dataframe(
                ranked_df,
                use_container_width=True,
                hide_index=True
            )

with st.expander("Model info"):
    st.write(f"**Model path:** `{MODEL_PATH}`")
    st.write(f"**Number of classes:** {len(label_map)}")
    st.write("**Classes:**")
    st.json(label_map)