# Intelligent Email Filtering System

An intelligent email filtering system that automatically classifies emails as Spam or Not Spam using Naïve Bayes classification algorithms.

## Features
- Email preprocessing pipeline (tokenization, stop-word removal, stemming)
- Three Naïve Bayes classifiers: Multinomial, Bernoulli, and Gaussian
- Real-time classification interface
- Confusion matrix visualization
- Accuracy comparison dashboard
- Model performance metrics

## Project Structure
```
├── data/                    # Dataset storage
├── models/                  # Trained models
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing
│   ├── training.py         # Model training
│   ├── evaluation.py       # Model evaluation
│   └── utils.py           # Utility functions
├── app.py                  # Streamlit web interface
├── train_model.py          # Training script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK data: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
4. Run training: `python train_model.py`
5. Launch app: `streamlit run app.py`

## Dataset
The system uses the SMS Spam Collection Dataset by default. You can also use the Enron Email Dataset by placing it in the `data/` directory.

## Usage
1. Train the model: `python train_model.py`
2. Start the web interface: `streamlit run app.py`
3. Enter email text in the web interface to get real-time classification

## Algorithms Compared
- **Multinomial Naïve Bayes**: Best for text classification with word counts
- **Bernoulli Naïve Bayes**: Suitable for binary/boolean features
- **Gaussian Naïve Bayes**: For continuous features (requires different feature representation)

## Performance Metrics
- Accuracy score
- Precision, Recall, F1-score
- Confusion matrix
- ROC curve (where applicable)