# 📰 Fake News Detection using Machine Learning

This project aims to **classify news articles as either FAKE or REAL** using Natural Language Processing (NLP) and machine learning techniques.  
It utilizes the **TF-IDF vectorization** method for text feature extraction and a **Linear Support Vector Classifier (LinearSVC)** for model training and prediction.

---

## 📊 Dataset
The dataset used is **fake_or_real_news.csv**, which contains the following columns:
- **id** — Unique identifier of each news article.  
- **title** — The title of the news.  
- **text** — The full text/content of the article.  
- **label** — Indicates whether the article is “FAKE” or “REAL”.

After encoding, the labels are converted as:
- FAKE → 0  
- REAL → 1  

---

## 🧠 Model Workflow

1. **Data Loading and Exploration**  
   The dataset is read using `pandas` and inspected for basic statistics.

2. **Label Encoding**  
   Replaced categorical labels (`FAKE`, `REAL`) with numerical values (`0`, `1`).

3. **Train-Test Split**  
   Split the dataset into 80% training and 20% testing subsets.

4. **Text Vectorization**  
   Used `TfidfVectorizer` to convert text data into numerical features while removing English stop words and ignoring overly frequent terms (`max_df=0.7`).

5. **Model Training**  
   Trained a **Linear Support Vector Classifier (LinearSVC)** model on the vectorized data.

6. **Evaluation**  
   Achieved an accuracy of approximately **93%** on the test set.

7. **Testing the Model**  
   - Predicts correctly whether a sample article is **real or fake**.

---

## 🧩 Technologies Used
- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn (sklearn)**
- **TF-IDF Vectorizer**
- **LinearSVC**

---

## 🚀 Results
- Model Accuracy: **93.05%**
- Efficiently distinguishes between fake and real news.
- Demonstrates the power of TF-IDF + Linear SVM for text classification.

---

## 💻 How to Run
```bash
# Clone the repository
git clone https://github.com/Kerollos-A/codSoft_Projects

# Navigate to the project directory
cd Fake_News_Detection

# Install dependencies
pip install numpy pandas scikit-learn

# Run the notebook
jupyter notebook Fake_News_Detection.ipynb
```

---

## 📈 Future Improvements
- Use deep learning (LSTM or BERT) for better text understanding.
- Integrate a live fake-news detection web app using **Flask** or **Streamlit**.
- Expand dataset for higher generalization accuracy.

---

## ✨ Author
**Kerollos Abdo**  
Certified Data Engineer | Machine Learning Enthusiast  
🔗 [GitHub Profile](https://github.com/Kerollos-A)
