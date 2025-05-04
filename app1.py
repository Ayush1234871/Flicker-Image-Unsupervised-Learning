import os
import re
import io
import pickle
import base64
import streamlit as st
import pandas as pd
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Optional: use gdown if you prefer automatic download
import gdown

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# === FILE SETUP ===
MODEL_FILENAME = "clustering_pipeline.pkl"
MODEL_URL = "https://drive.google.com/uc?id=14a9e2G-DLT5yZLgzqfr9CwaOoLgP4JLi"

# Try to download model if not already present
if not os.path.exists(MODEL_FILENAME):
    try:
        st.info("Downloading model file...")
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
    except Exception as e:
        st.error(f"Auto-download failed: {e}")

# === LOAD MODEL ===
try:
    with open(MODEL_FILENAME, "rb") as f:
        models = pickle.load(f)
    vectorizer = models['vectorizer']
    svd = models['svd']
    kmeans = models['kmeans']
    dbscan = models['dbscan']
    agg = models['agg']
except Exception as e:
    st.error("Failed to load model. Please ensure 'clustering_pipeline.pkl' is valid and in the app directory.")
    st.stop()

# === SETUP ===
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    stemmed = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(stemmed)

def predict_clusters(new_comments):
    new_cleaned = [clean_text(comment) for comment in new_comments]
    X_new_tfidf = vectorizer.transform(new_cleaned)
    X_new_lsa = svd.transform(X_new_tfidf)

    kmeans_labels = kmeans.predict(X_new_lsa)
    dbscan_labels = dbscan.fit_predict(X_new_lsa)
    agg_labels = agg.fit_predict(X_new_lsa) if len(new_comments) >= 20 else [-1] * len(new_comments)

    final_labels = []
    for i in range(len(new_comments)):
        votes = [kmeans_labels[i], dbscan_labels[i], agg_labels[i]]
        filtered_votes = [v for v in votes if v != -1]
        label = Counter(filtered_votes).most_common(1)[0][0] if filtered_votes else -1
        final_labels.append(label)

    return final_labels

# === STREAMLIT UI ===
st.set_page_config(page_title="Comment Cluster Predictor")
st.title("🧠 Comment Clustering Predictor")

mode = st.sidebar.radio("Choose input mode", ["Manual Comment", "Upload CSV"])

if mode == "Manual Comment":
    comment_input = st.text_area("Enter your comment(s):", placeholder="Separate multiple sentences with periods, question marks, etc.")

    if st.button("Predict Clusters"):
        if comment_input.strip():
            sentences = re.split(r'[.!?]+', comment_input)
            sentences = [s.strip() for s in sentences if s.strip()]
            labels = predict_clusters(sentences)
            temp_df = pd.DataFrame({'comment': sentences, 'cluster': labels})

            st.success("Prediction complete.")
            cluster_map = temp_df.groupby("cluster")["comment"].apply(list).to_dict()

            for cluster_id, comments in sorted(cluster_map.items()):
                st.markdown(f"### Cluster {cluster_id}")
                for comment in comments:
                    st.markdown(f"- {comment}")
        else:
            st.warning("Please enter at least one comment.")

elif mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'comment' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "comment" not in df.columns:
            st.error("CSV must contain a 'comment' column.")
        else:
            st.dataframe(df.head())

            if st.button("Predict Clusters"):
                labels = predict_clusters(df["comment"])
                df["cluster"] = labels

                st.success("Prediction complete.")
                cluster_map = df.groupby("cluster")["comment"].apply(list).to_dict()

                for cluster_id, comments in sorted(cluster_map.items()):
                    st.markdown(f"### Cluster {cluster_id}")
                    for comment in comments:
                        st.markdown(f"- {comment}")

                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="clustered_comments.csv">📥 Download Result CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Please upload a CSV file to begin.")
