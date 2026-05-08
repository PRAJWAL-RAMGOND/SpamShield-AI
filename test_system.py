#!/usr/bin/env python3
"""
Test script to verify the email filtering system works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import EmailPreprocessor, load_sms_dataset
from src.utils import create_sample_emails


def test_preprocessing():
    """Test the text preprocessing pipeline."""
    print("Testing preprocessing pipeline...")
    
    preprocessor = EmailPreprocessor()
    
    test_texts = [
        "WINNER!! You have won a free ticket to Bahamas! Call 12345",
        "Hi, are we still meeting tomorrow at 3pm?",
        "URGENT! Your bank account needs verification. Click here."
    ]
    
    for text in test_texts:
        processed = preprocessor.preprocess_text(text)
        print(f"\nOriginal: {text}")
        print(f"Processed: {processed}")
    
    print("\nPreprocessing test completed successfully!")


def test_sample_predictions():
    """Test predictions on sample emails."""
    print("\nTesting predictions on sample emails...")
    
    samples = create_sample_emails()
    
    for text, expected_label in samples[:3]:  # Test first 3 samples
        print(f"\nEmail: {text[:50]}...")
        print(f"Expected: {expected_label}")
        
        # Note: This would require trained models
        # For now, just show we can create the sample data
        pass
    
    print(f"\nCreated {len(samples)} sample emails for testing")


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\nTesting dataset loading...")
    
    try:
        texts, labels = load_sms_dataset('data/spam.csv')
        print(f"Dataset loaded successfully: {len(texts)} samples")
    except FileNotFoundError:
        print("Real dataset not found (expected for first run)")
        
        # Test with sample data
        from data.sample_data import create_sample_dataset
        df = create_sample_dataset()
        print(f"Sample dataset created: {len(df)} samples")
        print(f"Spam: {len(df[df['label'] == 'spam'])}")
        print(f"Ham: {len(df[df['label'] == 'ham'])}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("EMAIL FILTERING SYSTEM - TEST SUITE")
    print("=" * 60)
    
    test_preprocessing()
    test_sample_predictions()
    test_dataset_loading()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
    print("3. Train models: python train_model.py")
    print("4. Launch web interface: streamlit run app.py")


if __name__ == "__main__":
    main()