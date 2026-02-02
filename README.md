# Email Spam Detection: Precision-Focused NLP using Fast API

This project is a full-stack Machine Learning application that identifies spam emails using Natural Language Processing (NLP). It features a high-precision **Logistic Regression** model and a modern **FastAPI** web interface for real-time predictions.



## Project Overview
In spam detection, **Precision is the priority**. Misclassifying a critical legitimate email as spam (False Positive) is a high-cost error. This project successfully optimized for precision, achieving a near-perfect result with only **1 false positive** in the best-performing model.

## 🎯 Project Objectives
- **Accurate Classification:** Distinguish between Spam and Ham using NLP.
- **Precision Optimization:** Minimize False Positives to protect important user emails.
- **Model Comparison:** Evaluate 8 different algorithms to find the most robust solution.

## 🛠️ Tech Stack
- **Languages:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, NLTK, BeautifulSoup, Joblib
- **Techniques:** TF-IDF Vectorization, Porter Stemming, Regex Cleaning, Emoji Demojization

### Key Highlights:
- **Winning Model:** Logistic Regression (97.19% Accuracy).
- **Vectorization:** CountVectorizer (Bag of Words).
- **Deployment:** FastAPI with Jinja2 HTML templates.
- **Custom NLP Pipeline:** Handles emojis, removes URLs, and implements Porter Stemming.

---

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | False Positives |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **97.19%** | **97.25%** | **97.19%** | **1** |
| **XGBoost Classifier** | 96.32% | 96.27% | 96.32% | 6 |
| **Multinomial NB** | 95.64% | 96.24% | 95.64% | 37 |
| **Random Forest** | 91.28% | 92.08% | 91.28% | 0 |



---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/darshil1995/Spam-Detection-Project.git](https://github.com/darshil1995/Spam-Detection-Project.git)
cd Spam-Detection-Project/ModelProjects

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the Web Application

```bash
python main.py

```

After running the command, open your browser and navigate to:

👉 **http://127.0.0.1:8001**

---

## 📂 Project Structure

```text
├── ModelProjects/
│   ├── main.py              # FastAPI Server & Cleaning Logic
│   ├── models/              # Saved .joblib files
│   │   ├── spam_model.joblib
│   │   └── count_vectorizer.joblib
│   └── templates/           # UI Files
│       └── index.html       # Web Interface
├── data/                    # Dataset (CSV)
├── notebooks/               # Training & EDA Notebook
└── requirements.txt         # Project Libraries

```

---

## 🧪 How it Works

1. **Input:** You paste a raw email message into the web interface.
2. **Preprocessing:** The API cleans the text using the exact same pipeline used in training (lowercase conversion, URL removal, emoji demojization, and Porter Stemming).
3. **Vectorization:** The cleaned text is converted into numerical counts using the pre-trained `CountVectorizer`.
4. **Prediction:** The Logistic Regression model classifies the message as **Spam** or **Ham**.

## 📝 License

This project is licensed under the MIT License.
