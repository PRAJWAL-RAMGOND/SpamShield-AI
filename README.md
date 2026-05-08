# 🛡️ SpamShield AI
### Intelligent Email Filtering System Using Naïve Bayes Classification

SpamShield AI is a Machine Learning-based spam email detection system that automatically classifies emails/messages as **Spam** or **Not Spam (Ham)** using the **Naïve Bayes Classification Algorithm**.

The project applies Natural Language Processing (NLP) techniques such as text preprocessing, tokenization, stop-word removal, stemming, and vectorization to analyze textual email data and predict spam probability with high accuracy.

---

# 📌 Project Overview

Email spam has become one of the major cybersecurity and communication challenges in modern digital systems. SpamShield AI addresses this problem by implementing intelligent filtering techniques using Machine Learning.

This project compares multiple Naïve Bayes algorithms:

- Multinomial Naïve Bayes
- Bernoulli Naïve Bayes
- Gaussian Naïve Bayes

The system preprocesses email text, extracts features using TF-IDF/Bag-of-Words, trains classification models, and predicts whether a message is spam or legitimate.

---

# 🎯 Objectives

- Detect spam emails/messages automatically
- Implement Naïve Bayes classification techniques
- Compare performance of different Naïve Bayes models
- Build a real-time spam prediction interface
- Visualize model performance using confusion matrix and accuracy metrics

---

# 🧠 Module Concepts Covered

- Bayes Theorem
- Conditional Probability
- Naïve Bayes Classification
- Feature Independence Assumption
- Natural Language Processing (NLP)
- Text Preprocessing
- TF-IDF & Bag-of-Words
- Machine Learning Model Evaluation

---

# 🚀 Features

✅ Email/Text preprocessing pipeline  
✅ Spam probability prediction  
✅ Real-time spam classification  
✅ Confusion matrix visualization  
✅ Accuracy comparison dashboard  
✅ Multiple Naïve Bayes algorithm comparison  
✅ Clean and user-friendly interface  
✅ Fast prediction system  

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core Programming |
| Scikit-learn | Machine Learning |
| Pandas | Data Handling |
| NumPy | Numerical Operations |
| NLTK | NLP Preprocessing |
| Matplotlib | Visualization |
| Seaborn | Graphs & Heatmaps |
| Streamlit / Flask | Web Interface |

---

# 📂 Project Structure

```bash
SpamShield-AI/
│
├── dataset/
│   └── spam.csv
│
├── models/
│   └── trained_model.pkl
│
├── notebooks/
│   └── model_training.ipynb
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# 📊 Dataset Used

## SMS Spam Collection Dataset

A publicly available dataset containing labeled SMS messages categorized as:

- Spam
- Ham (Not Spam)

Dataset Sources:
- UCI Machine Learning Repository
- Kaggle

---

# ⚙️ System Workflow

```text
Input Email/Text
        ↓
Text Preprocessing
(Tokenization, Stop-word Removal, Stemming)
        ↓
Feature Extraction
(TF-IDF / Bag-of-Words)
        ↓
Naïve Bayes Training
        ↓
Spam Prediction
        ↓
Result Visualization
```

---

# 🔄 Text Preprocessing Steps

The system performs several NLP preprocessing operations:

### 1. Tokenization
Splits text into individual words/tokens.

### 2. Stop-word Removal
Removes common words like:
- the
- is
- are
- and

### 3. Stemming
Converts words into root form:
- running → run
- playing → play

### 4. Vectorization
Transforms text into numerical features using:
- TF-IDF Vectorizer
- Bag-of-Words

---

# 🤖 Machine Learning Models Used

## 1️⃣ Multinomial Naïve Bayes

Best suited for text classification problems using word frequencies.

### Advantages
- Fast
- Accurate for NLP tasks
- Works well with TF-IDF

---

## 2️⃣ Bernoulli Naïve Bayes

Works with binary feature vectors.

### Advantages
- Efficient for binary occurrence data
- Good for shorter texts

---

## 3️⃣ Gaussian Naïve Bayes

Assumes features follow Gaussian distribution.

### Advantages
- Simple implementation
- Useful for continuous data

---

# 📈 Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 📉 Confusion Matrix

The confusion matrix helps visualize:

| Actual / Predicted | Spam | Not Spam |
|-------------------|------|-----------|
| Spam | TP | FN |
| Not Spam | FP | TN |

---

# ▶️ Installation & Setup

## Step 1: Clone Repository

```bash
git clone https://github.com/Preethu26/SpamShield-AI.git
```

## Step 2: Navigate to Project Folder

```bash
cd SpamShield-AI
```

## Step 3: Create Virtual Environment

```bash
python -m venv venv
```

## Step 4: Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Linux/Mac
```bash
source venv/bin/activate
```

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

## Using Flask

```bash
python app.py
```

## Using Streamlit

```bash
streamlit run app.py
```

---

# 🖥️ Sample Output

| Input Message | Prediction |
|---------------|------------|
| "Congratulations! You won a free iPhone" | Spam |
| "Meeting scheduled at 10 AM tomorrow" | Not Spam |

---

# 📸 Screenshots

## Home Page
(Add Screenshot Here)

## Prediction Result
(Add Screenshot Here)

## Accuracy Dashboard
(Add Screenshot Here)

---

# 📚 Advantages of SpamShield AI

- High spam detection accuracy
- Fast classification
- Lightweight ML model
- Easy deployment
- User-friendly interface
- Scalable for real-world applications

---

# 🔮 Future Enhancements

- Deep Learning integration
- Email attachment scanning
- Multi-language spam detection
- Cloud deployment
- Real-time email integration
- Advanced phishing detection

---

# 🧪 Results

| Algorithm | Accuracy |
|-----------|----------|
| Multinomial NB | 98% |
| Bernoulli NB | 96% |
| Gaussian NB | 91% |

> Multinomial Naïve Bayes produced the best performance for text classification.

---

# 👨‍💻 Author

## Preetham KP

Machine Learning & Web Development Enthusiast

GitHub:
https://github.com/Preethu26

Project Repository:
https://github.com/Preethu26/SpamShield-AI

---

# 📄 License

This project is licensed under the MIT License.

---

# 🙌 Acknowledgements

- Scikit-learn Documentation
- NLTK Documentation
- UCI ML Repository
- Kaggle Datasets
- Open Source Community

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
📢 Share with others  

---

# 📬 Contact

For queries or collaborations:

- GitHub: https://github.com/Preethu26

---

## 💡 “Turning Intelligent Filtering into Smarter Communication.”
