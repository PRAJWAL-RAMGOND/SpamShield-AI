#!/usr/bin/env python3
"""
Main training script for the email filtering system.
Trains all Naïve Bayes models and saves them to disk.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training import train_and_save_model
from src.evaluation import evaluate_models


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("INTELLIGENT EMAIL FILTERING SYSTEM")
    print("=" * 60)
    
    # Check for dataset
    dataset_path = 'data/spam.csv'
    if not os.path.exists(dataset_path):
        print(f"\nDataset not found at {dataset_path}")
        print("Please download the SMS Spam Collection dataset and place it in the data/ directory.")
        print("You can download it from: https://www.kaggle.com/uciml/sms-spam-collection-dataset")
        print("\nUsing sample data for demonstration...")
    
    try:
        # Train models
        trainer = train_and_save_model(dataset_path)
        
        # Evaluate models
        evaluator = evaluate_models(trainer)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Launch the web interface: streamlit run app.py")
        print("2. Test the system with real emails")
        print("3. Check the reports/ directory for detailed evaluation")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        print("3. Check if dataset exists at data/spam.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()