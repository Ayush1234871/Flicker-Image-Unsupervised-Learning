# 🖼️ Flicker Image Comment Clustering App

A Streamlit-based NLP app that clusters **text comments** (e.g., from Flicker image feedback) using **KMeans**, **DBSCAN**, and **Agglomerative Clustering**. Designed to handle large datasets (e.g., 150,000+ comments) with robust preprocessing and export functionality.

---

## 🚀 Features

- 📂 Upload CSV files containing text comments
- 📝 Enter individual comments manually
- ⚙️ Intelligent preprocessing:
  - Lowercasing
  - Removing stopwords
  - Lemmatization
  - Handling missing/duplicate values
- ⬇️ Download clustered results as CSV

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **NLP**: NLTK, spaCy
- **Vectorization**: TF-IDF
- **Clustering Algorithms**: 
  - KMeans (scikit-learn)
  - DBSCAN (scikit-learn)
  - Agglomerative Clustering (scikit-learn)
- **Others**: Pandas, NumPy

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flicker-image-clustering.git
   cd flicker-image-clustering
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
flicker-image-clustering/
├── app1.py
├── comment_clustering
    ├── requirements.txt
└── output.csv
└── main.ipynb
```

---

## 🖼️ Use Case

Originally designed to cluster and analyze feedback/comments on **Flicker images**, this tool helps discover:
- Comment themes
- User sentiment groups
- Spam vs. genuine content clusters

It’s also adaptable for Amazon reviews, tweets, or any large text corpus.

---

