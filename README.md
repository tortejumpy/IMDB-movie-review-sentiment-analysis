# 🎬 IMDb Movie Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP) & Machine Learning** project that classifies IMDb movie reviews into **positive** or **negative** sentiment. The project focuses on **robust text preprocessing, feature engineering, model comparison, and error analysis**, closely simulating a real-world sentiment analysis pipeline.

---

## 📌 Project Overview

Online platforms like IMDb receive thousands of user reviews daily. Manually understanding audience sentiment is inefficient and error-prone. This project builds a scalable sentiment classification system that automatically interprets user opinions using classical NLP and machine learning techniques.

**Key Goal:**

> Predict whether a movie review expresses **positive** or **negative** sentiment with high accuracy and interpretability.

---

## 🧠 Business Use Case

* Understand audience perception of movies at scale
* Support data-driven marketing and content decisions
* Monitor sentiment trends without manual review analysis

---

## 📂 Dataset

* **Source:** IMDb Movie Reviews Dataset
* **Size:** 50,000 reviews
* **Classes:**

  * Positive: 25,000
  * Negative: 25,000
* **Target Variable:** `sentiment` (positive / negative)

---

## 🔍 Phase 1: Data Exploration & Analysis

### ✔ Key Steps

* Checked class balance (perfectly balanced dataset)
* Analyzed review length (characters & word count)
* Identified long-review outliers using IQR method
* Compared review statistics across sentiments

### 🔎 Key Insights

* Review lengths vary significantly (from 1 word to 2,400+ words)
* Negative reviews tend to be slightly more complex linguistically
* Dataset balance allowed unbiased model evaluation

---

## 🧹 Phase 2: Text Preprocessing

### ✔ Cleaning Pipeline

* Removed HTML tags, URLs, punctuation, and numbers
* Converted text to lowercase
* Tokenized reviews
* Removed stopwords
* Applied **lemmatization** (preferred over stemming)

> Negation words (e.g., *not*, *never*) were preserved to retain sentiment meaning.

---

## 🛠 Phase 3: Feature Engineering

### 🔹 1. Semantic Features

* **TF-IDF Vectorization**

  * Unigrams, bigrams, and combined n-grams
  * Max features: 1000–2500
* **Word2Vec Embeddings** (Skip-gram)

  * 100-dimensional word vectors
  * Averaged to form document vectors

### 🔹 2. Handcrafted Textual Features

* **Basic:** word count, sentence count, avg word length
* **Lexical:** lexical diversity, hapax ratio
* **Readability:** Flesch, SMOG, Coleman–Liau
* **Style:** stopword ratio, punctuation density

### 🔹 3. Dimensionality Reduction

* **Truncated SVD** for TF-IDF (100 components)
* **PCA** for Word2Vec (50 components)

---

## 🧩 Feature Set Variants

| Feature Set | Description                                 |
| ----------- | ------------------------------------------- |
| FS1         | TF-IDF only                                 |
| FS2         | Textual features only                       |
| FS3         | TF-IDF + Textual                            |
| FS4         | Reduced TF-IDF + Reduced Word2Vec + Textual |
| FS5         | All features combined                       |

---

## 🤖 Phase 4: Model Development

### Models Trained

* Logistic Regression
* Naive Bayes
* Linear SVM
* Random Forest

### Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Cross-validation accuracy

---

## 🏆 Model Performance Summary

| Model                   | Feature Set      | Accuracy   |
| ----------------------- | ---------------- | ---------- |
| **Logistic Regression** | **TF-IDF**       | **87.48%** |
| Logistic Regression     | TF-IDF + Textual | 86.76%     |
| SVM (Linear)            | TF-IDF           | 84.69%     |
| Naive Bayes             | TF-IDF           | 84.26%     |
| Random Forest           | TF-IDF           | 81.09%     |

✔ **Best Model:** Logistic Regression with TF-IDF

---

## 🔎 Phase 5: Error Analysis

### Error Breakdown (Best Model: LR + TF-IDF)

* **Total Errors:** 12.52%
* **False Positives:** 54.7%
* **False Negatives:** 45.3%

### Class-wise Error Rate

* Negative Reviews: **13.7%**
* Positive Reviews: **11.3%**

### Key Insight

> Negative sentiment is harder to detect due to sarcasm, mixed opinions, and subtle wording.

---

## 📊 Visualizations Included

* Sentiment distribution plots
* Review length histograms & boxplots
* TF-IDF vs Word2Vec variance curves
* Feature correlation heatmaps
* Model accuracy & training time comparison
* Confusion matrices & error distribution charts

---

## 🧰 Tech Stack

* **Programming:** Python
* **NLP:** NLTK, spaCy, TextBlob, Gensim
* **ML:** scikit-learn
* **Vectorization:** TF-IDF, Word2Vec
* **Visualization:** matplotlib, seaborn
* **Notebook:** Jupyter

---

## 🚀 Key Learnings

* TF-IDF remains extremely powerful for classical NLP tasks
* Logistic Regression performs best on high-dimensional sparse text data
* Handcrafted linguistic features improve interpretability but not accuracy
* Error analysis is critical for understanding real-world limitations

---

## 🔮 Future Improvements

* Integrate **BERT / Transformer-based models** for contextual understanding
* Handle sarcasm and negation using attention mechanisms
* Deploy model as an API or web application

---

## 📌 Conclusion

This project demonstrates a **production-style NLP pipeline**, from raw text to model evaluation and error analysis. It highlights strong feature engineering, model selection, and analytical reasoning — making it highly suitable for **Data Scientist and Machine Learning Engineer roles**.

---

⭐ *If you found this project useful, feel free to star the repository!*
