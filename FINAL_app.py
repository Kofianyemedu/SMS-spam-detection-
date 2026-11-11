"""
SMS Spam Detection - Streamlit App
Group 6 - MSc Business Analytics Text Analytics Project

Brief requirement: "Use Python with Streamlit" + "Deploy the project on
free server and share the link: Streamlit Community Cloud".

Structure mirrors the project brief's required stages:
  Corpora Viewer -> Preprocessing -> Exploratory Text Analytics ->
  Text Representation & Models -> Live Prediction
"""

import html
import re
import ssl
from collections import Counter


import numpy as np
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

DATA_PATH = "SMSSpamCollection"
RANDOM_STATE = 42

st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="wide")

# --- NLTK setup (same SSL workaround used in the PyCharm project) ---
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------
# PREPROCESSING (merged in from the former preprocessing.py)
# Brief Structure items: "Data cleaning; Regular Expression" + "Text preprocessing"
# ---------------------------------------------------------------
def clean_text(text):
    text = html.unescape(text)                                     # decode entities: &lt;#&gt; -> <#>
    text = text.lower()                                            # case normalization
    text = re.sub(r'<#>', ' num ', text)                           # dataset's masked-number placeholder
    text = re.sub(r'<url>', ' url ', text)                         # dataset's masked-URL placeholder
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)              # URLs -> placeholder
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', ' email ', text)  # emails -> placeholder
    text = re.sub(r'[£$€]', ' currency ', text)                    # currency symbols -> placeholder
    text = re.sub(r'\b\d{5,}\b', ' longnum ', text)                 # long digit strings -> placeholder
    text = re.sub(r'\d+', ' num ', text)                            # remaining digits -> placeholder
    text = re.sub(r'[^a-z\s]', ' ', text)                           # strip remaining punctuation
    text = re.sub(r'\s+', ' ', text).strip()                        # normalize whitespace
    return text


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def run_pipeline(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    df = df.drop_duplicates().reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["preprocessed_text"] = df["clean_text"].apply(preprocess_text)
    return df


# ---------------------------------------------------------------
# CACHED DATA / MODEL LOADING
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    df = run_pipeline(DATA_PATH)
    return df


@st.cache_resource(show_spinner="Training models...")
def train_models(_df):
    X = _df["preprocessed_text"]
    y = _df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    tfidf = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    svm = LinearSVC(random_state=RANDOM_STATE)
    svm.fit(X_train_tfidf, y_train)

    return {
        "tfidf": tfidf,
        "nb": nb,
        "svm": svm,
        "X_test_tfidf": X_test_tfidf,
        "y_test": y_test,
    }


def compute_metrics(y_test, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_score),
    }


def top_n_words(texts, n=20):
    all_words = " ".join(texts).split()
    return Counter(all_words).most_common(n)


def top_n_bigrams(texts, n=15):
    all_bigrams = []
    for t in texts:
        tokens = t.split()
        all_bigrams.extend(ngrams(tokens, 2))
    counted = Counter(all_bigrams).most_common(n)
    return [(" ".join(bg), c) for bg, c in counted]


# ---------------------------------------------------------------
# PAGE 1 - CORPORA VIEWER
# ---------------------------------------------------------------
def page_overview(df):
    st.subheader("Corpora Viewer")
    st.markdown(
        """
        **Dataset:** UCI SMS Spam Collection
        **Task:** Binary text classification — `ham` (legitimate) vs `spam`
        **Representation:** TF-IDF | **Models:** Naive Bayes + SVM
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total messages", len(df))
    col2.metric("Ham", int((df["label"] == "ham").sum()))
    col3.metric("Spam", int((df["label"] == "spam").sum()))

    if st.checkbox("Show raw dataset", key="show_raw_dataset"):
        st.dataframe(df[["label", "text"]], use_container_width=True)

    if st.checkbox("Show preprocessed dataset", key="show_preprocessed_dataset"):
        st.dataframe(
            df[["label", "text", "clean_text", "preprocessed_text"]],
            use_container_width=True,
        )


# ---------------------------------------------------------------
# PAGE 2 - TEXT PREPROCESSING
# ---------------------------------------------------------------
def page_preprocessing(df):
    st.subheader("Text Preprocessing Pipeline")
    st.markdown(
        """
        Steps applied (per project brief Structure + lecture slides L2/L3):
        1. **Data cleaning (Regular Expressions):** lowercase, decode HTML entities,
           mask URLs/emails/currency symbols/digits with placeholder tokens,
           strip remaining punctuation.
        2. **Tokenization:** NLTK `word_tokenize`.
        3. **Stopword removal:** NLTK English stopword list.
        4. **Lemmatization:** NLTK `WordNetLemmatizer`.
        """
    )

    sample_idx = st.slider("Pick a message to inspect", 0, len(df) - 1, 0, key="sample_idx")
    row = df.iloc[sample_idx]

    st.write("**Label:**", row["label"])
    st.write("**Original text:**", row["text"])
    st.write("**After regex cleaning:**", row["clean_text"])
    st.write("**After tokenize + stopwords + lemmatize:**", row["preprocessed_text"])


# ---------------------------------------------------------------
# PAGE 3 - EXPLORATORY TEXT ANALYTICS
# ---------------------------------------------------------------
def page_eda(df):
    st.subheader("Exploratory Text Analytics")

    ham = df[df["label"] == "ham"]
    spam = df[df["label"] == "spam"]

    # Class distribution + length
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        df["label"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_title("Class Distribution")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        df_len = df.copy()
        df_len["word_len"] = df_len["text"].str.split().str.len()
        fig, ax = plt.subplots(figsize=(4, 3))
        for label, color in zip(["ham", "spam"], ["#4C72B0", "#DD8452"]):
            subset = df_len[df_len["label"] == label]
            ax.hist(subset["word_len"], bins=25, alpha=0.6, label=label, color=color)
        ax.set_title("Message Length (words)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Top words
    st.markdown("**Top 20 words by class (post-cleaning)**")
    ham_top = top_n_words(ham["preprocessed_text"], 20)
    spam_top = top_n_words(spam["preprocessed_text"], 20)
    col1, col2 = st.columns(2)
    for col, data, title, color in zip(
        [col1, col2], [ham_top, spam_top], ["HAM", "SPAM"], ["#4C72B0", "#DD8452"]
    ):
        with col:
            fig, ax = plt.subplots(figsize=(5, 5))
            words, counts = zip(*sorted(data, key=lambda x: x[1]))
            ax.barh(words, counts, color=color)
            ax.set_title(f"Top words - {title}")
            st.pyplot(fig)
            plt.close(fig)

    # Word clouds
    st.markdown("**Word clouds by class**")
    col1, col2 = st.columns(2)
    for col, subset, title in zip([col1, col2], [ham, spam], ["HAM", "SPAM"]):
        with col:
            text_blob = " ".join(subset["preprocessed_text"])
            wc = WordCloud(width=500, height=350, background_color="white",
                            colormap="viridis").generate(text_blob)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title)
            st.pyplot(fig)
            plt.close(fig)

    # Surface signals
    st.markdown("**Surface signal comparison**")

    def digit_ratio(t):
        return sum(c.isdigit() for c in t) / max(len(t), 1)

    def upper_word_count(t):
        return sum(1 for w in t.split() if w.isupper() and len(w) > 1)

    def has_currency(t):
        return bool(re.search(r"[£$€]", t))

    signals = pd.DataFrame({
        "label": df["label"],
        "digit_ratio": df["text"].apply(digit_ratio),
        "upper_word_count": df["text"].apply(upper_word_count),
        "exclaim_count": df["text"].str.count("!"),
        "has_currency": df["text"].apply(has_currency),
    })
    summary = signals.groupby("label").agg(
        avg_digit_ratio=("digit_ratio", "mean"),
        avg_upper_words=("upper_word_count", "mean"),
        avg_exclaim_count=("exclaim_count", "mean"),
        pct_with_currency=("has_currency", "mean"),
    ).round(3)
    st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------------
# PAGE 4 - MODEL COMPARISON
# ---------------------------------------------------------------
def page_models(df):
    st.subheader("Text Representation (TF-IDF) & Model Comparison")

    trained = train_models(df)
    tfidf = trained["tfidf"]
    nb, svm = trained["nb"], trained["svm"]
    X_test_tfidf, y_test = trained["X_test_tfidf"], trained["y_test"]

    st.write(f"TF-IDF vocabulary size: **{len(tfidf.get_feature_names_out())}**")

    nb_pred = nb.predict(X_test_tfidf)
    nb_score = nb.predict_proba(X_test_tfidf)[:, 1]
    svm_pred = svm.predict(X_test_tfidf)
    svm_score = svm.decision_function(X_test_tfidf)

    nb_metrics = compute_metrics(y_test, nb_pred, nb_score)
    svm_metrics = compute_metrics(y_test, svm_pred, svm_score)
    results_df = pd.DataFrame({"Naive Bayes": nb_metrics, "SVM": svm_metrics}).T
    st.dataframe(results_df.round(4), use_container_width=True)

    best = results_df["F1-score"].idxmax()
    st.success(f"Best model by F1-score: **{best}**")

    col1, col2 = st.columns(2)
    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, pred, title in zip(axes, [nb_pred, svm_pred], ["Naive Bayes", "SVM"]):
            cm = confusion_matrix(y_test, pred)
            ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(
                ax=ax, colorbar=False, cmap="Blues"
            )
            ax.set_title(title)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for score, name in [(nb_score, "Naive Bayes"), (svm_score, "SVM")]:
            fpr, tpr, _ = roc_curve(y_test, score)
            auc_val = roc_auc_score(y_test, score)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------
# PAGE 5 - LIVE PREDICTION
# ---------------------------------------------------------------
def page_predict(df):
    st.subheader("Live SMS Spam Prediction")

    trained = train_models(df)
    tfidf, nb, svm = trained["tfidf"], trained["nb"], trained["svm"]

    model_choice = st.radio("Choose model", ["SVM (recommended)", "Naive Bayes"], horizontal=True)

    if "user_sms" not in st.session_state:
        st.session_state.user_sms = ""

    user_input = st.text_area("Enter an SMS message", key="user_sms", height=120)

    col1, col2, _ = st.columns([1, 1, 4])
    predict_clicked = col1.button("Predict", width='stretch')
    if col2.button("Clear", width='stretch'):
       ### st.session_state.user_sms = ""

        st.rerun()

    if predict_clicked:
        if not user_input.strip():
            st.warning("Please enter a message first.")
        else:
            processed = preprocess_text(clean_text(user_input))
            vec = tfidf.transform([processed])

            if model_choice.startswith("SVM"):
                pred = svm.predict(vec)[0]
                score = svm.decision_function(vec)[0]
                confidence_note = f"Decision score: {score:.2f} (further from 0 = more confident)"
            else:
                pred = nb.predict(vec)[0]
                proba = nb.predict_proba(vec)[0][1]
                confidence_note = f"Spam probability: {proba:.2%}"

            label = "SPAM" if pred == 1 else "HAM"
            if pred == 1:
                st.error(f"🚫 Prediction: **{label}**")
            else:
                st.success(f"✅ Prediction: **{label}**")
            st.caption(confidence_note)
            st.caption(f"Preprocessed text used for prediction: `{processed}`")


# ---------------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------------
def main():
    st.title("📱 SMS Spam Detection")
    st.caption("Group 6 — MSc Business Analytics Text Analytics Project")

    df = load_data()

    pages = {
        "Corpora Viewer": page_overview,
        "Text Preprocessing": page_preprocessing,
        "Exploratory Text Analytics": page_eda,
        "TF-IDF & Model Comparison": page_models,
        "Live Prediction": page_predict,
    }
    selection = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[selection](df)


if __name__ == "__main__":
    main()
