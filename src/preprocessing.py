"""
Text preprocessing module for email filtering system.
Includes tokenization, stop-word removal, stemming, and feature extraction.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np


class EmailPreprocessor:
    """Preprocesses email text for spam classification."""
    
    def __init__(self, use_stemming=True, use_tfidf=True):
        """
        Initialize the preprocessor.
        
        Args:
            use_stemming (bool): Whether to apply stemming
            use_tfidf (bool): Whether to use TF-IDF (True) or Bag-of-Words (False)
        """
        self.use_stemming = use_stemming
        self.use_tfidf = use_tfidf
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
    def clean_text(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        if not self.use_stemming:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming
        tokens = self.stem_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataset(self, texts):
        """
        Preprocess a list of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def fit_vectorizer(self, texts):
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts (list): List of training texts
            
        Returns:
            sklearn vectorizer: Fitted vectorizer
        """
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=5000)
        else:
            self.vectorizer = CountVectorizer(max_features=5000)
        
        self.vectorizer.fit(texts)
        return self.vectorizer
    
    def transform_texts(self, texts):
        """
        Transform texts to feature vectors.
        
        Args:
            texts (list): List of texts
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted first. Call fit_vectorizer()")
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts (list): List of texts
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        self.fit_vectorizer(texts)
        return self.transform_texts(texts)


def load_sms_dataset(filepath='data/spam.csv'):
    """
    Load and prepare the SMS Spam Collection dataset.
    
    Args:
        filepath (str): Path to dataset file
        
    Returns:
        tuple: (texts, labels)
    """
    try:
        # Try to load the dataset
        df = pd.read_csv(filepath, encoding='latin-1')
        
        # The SMS dataset typically has columns: v1 (label), v2 (message)
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
        elif 'label' in df.columns and 'text' in df.columns:
            # Already in correct format
            pass
        else:
            # Try to infer columns
            df.columns = ['label', 'text', 'unused1', 'unused2', 'unused3']
            df = df[['label', 'text']]
        
        # Convert labels to binary (spam=1, ham=0)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Handle missing values
        df = df.dropna()
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        return texts, labels
        
    except FileNotFoundError:
        print(f"Dataset not found at {filepath}. Using sample data.")
        # Return sample data for demonstration
        sample_texts = [
            "WINNER!! You have won a free ticket to Bahamas! Call 12345",
            "Hi, are we still meeting tomorrow at 3pm?",
            "URGENT! Your bank account needs verification. Click here.",
            "Hey, just checking in. How are you doing?",
            "Congratulations! You've been selected for a free iPhone."
        ]
        sample_labels = [1, 0, 1, 0, 1]  # 1=spam, 0=ham
        
        return sample_texts, sample_labels


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = EmailPreprocessor()
    
    # Sample text
    sample_text = "WINNER!! You have won a free ticket to Bahamas! Call 12345 now!"
    
    # Test preprocessing
    processed = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    
    # Test on sample dataset
    texts, labels = load_sms_dataset()
    print(f"\nLoaded {len(texts)} samples")
    print(f"Spam samples: {sum(labels)}")
    print(f"Ham samples: {len(labels) - sum(labels)}")