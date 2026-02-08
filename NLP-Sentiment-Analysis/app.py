import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils.text_preprocessing import clean_text, preprocess

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Sentiment Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    model = joblib.load("notebooks/sentiment_lr_balanced.pkl")
    vectorizer = joblib.load("notebooks/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_pipeline()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Live Prediction", "Batch Analysis", "Sentiment Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "NLP Sentiment Analysis\n\n"
    "• TF-IDF\n"
    "• SVM / Logistic Regression\n"
    "• Real-time Prediction"
)

# --------------------------------------------------
# PAGE 1: LIVE PREDICTION
# --------------------------------------------------
if page == "Live Prediction":
    st.title("📝 Live Customer Review Sentiment")

    review = st.text_area(
        "Enter a customer review",
        height=150,
        placeholder="Type or paste a customer review here..."
    )

    if st.button("🔍 Analyze Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a review")
        else:
            cleaned = preprocess(clean_text(review))
            X = vectorizer.transform([cleaned])
            prediction = model.predict(X)[0]

            st.subheader("Prediction Result")
            if prediction == "Positive":
                st.success("😊 Positive Sentiment")
            elif prediction == "Negative":
                st.error("😠 Negative Sentiment")
            else:
                st.warning("😐 Neutral Sentiment")

# --------------------------------------------------
# PAGE 2: BATCH ANALYSIS
# --------------------------------------------------
elif page == "Batch Analysis":
    st.title("📂 Batch Review Sentiment Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV with a 'Review Text' column",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Review Text" not in df.columns:
            st.error("CSV must contain a 'Review Text' column")
        else:
            df["processed"] = df["Review Text"].astype(str).apply(
                lambda x: preprocess(clean_text(x))
            )

            X_batch = vectorizer.transform(df["processed"])
            df["Predicted Sentiment"] = model.predict(X_batch)

            st.success("Analysis completed!")
            st.dataframe(df.head(20))

            # Sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Predicted Sentiment", data=df, ax=ax)
            st.pyplot(fig)

# --------------------------------------------------
# PAGE 3: SENTIMENT INSIGHTS
# --------------------------------------------------
elif page == "Sentiment Insights":
    st.title("📈 Sentiment Insights Dashboard")

    df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
    df = df.dropna(subset=["Review Text"])

    df["sentiment"] = df["Rating"].apply(
        lambda x: "Positive" if x >= 4 else "Neutral" if x == 3 else "Negative"
    )

    st.subheader("Overall Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="sentiment", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Sentiment vs Rating Heatmap")
    heatmap_data = pd.crosstab(df["Rating"], df["sentiment"])
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
