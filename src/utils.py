"""
Utility functions for the email filtering system.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple


def ensure_directory(directory: str) -> str:
    """
    Ensure a directory exists.
    
    Args:
        directory (str): Directory path
        
    Returns:
        str: Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def load_model(model_name: str, models_dir: str = 'models') -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_name (str): Name of the model (multinomial_nb, bernoulli_nb, gaussian_nb)
        models_dir (str): Directory containing models
        
    Returns:
        Any: Loaded model
    """
    model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)


def load_preprocessor(models_dir: str = 'models') -> Any:
    """
    Load the preprocessor from disk.
    
    Args:
        models_dir (str): Directory containing models
        
    Returns:
        Any: Loaded preprocessor
    """
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    return joblib.load(preprocessor_path)


def load_metrics(models_dir: str = 'models') -> Dict[str, Any]:
    """
    Load model metrics from disk.
    
    Args:
        models_dir (str): Directory containing models
        
    Returns:
        Dict[str, Any]: Loaded metrics
    """
    metrics_path = os.path.join(models_dir, 'metrics.json')
    
    if not os.path.exists(metrics_path):
        return {}
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def predict_email(text: str, model_name: str = 'multinomial_nb', 
                  models_dir: str = 'models') -> Tuple[float, str]:
    """
    Predict if an email is spam.
    
    Args:
        text (str): Email text
        model_name (str): Name of model to use
        models_dir (str): Directory containing models
        
    Returns:
        Tuple[float, str]: (probability, label)
    """
    try:
        # Load preprocessor and model
        preprocessor = load_preprocessor(models_dir)
        model = load_model(model_name, models_dir)
        
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        # Transform to features
        features = preprocessor.transform_texts([processed_text])
        
        # For Bernoulli NB, need to binarize features
        if model_name == 'bernoulli_nb':
            features = (features > 0).astype(int)
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            spam_probability = proba[1]  # Probability of spam (class 1)
        else:
            # If model doesn't have predict_proba, use predict
            prediction = model.predict(features)[0]
            spam_probability = float(prediction)
        
        # Determine label
        label = "Spam" if spam_probability >= 0.5 else "Not Spam"
        
        return spam_probability, label
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def get_model_info(models_dir: str = 'models') -> Dict[str, Any]:
    """
    Get information about available models.
    
    Args:
        models_dir (str): Directory containing models
        
    Returns:
        Dict[str, Any]: Model information
    """
    info = {
        'available_models': [],
        'best_model': None,
        'best_accuracy': 0
    }
    
    # Load metrics
    metrics = load_metrics(models_dir)
    
    if metrics:
        info['available_models'] = list(metrics.keys())
        
        # Find best model
        for model_name, model_metrics in metrics.items():
            accuracy = model_metrics.get('accuracy', 0)
            if accuracy > info['best_accuracy']:
                info['best_accuracy'] = accuracy
                info['best_model'] = model_name
    
    return info


def format_probability(probability: float) -> str:
    """
    Format probability as percentage.
    
    Args:
        probability (float): Probability value (0-1)
        
    Returns:
        str: Formatted percentage
    """
    return f"{probability * 100:.2f}%"


def get_feature_importance(model, preprocessor, top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Get top features for Naïve Bayes models.
    
    Args:
        model: Trained Naïve Bayes model
        preprocessor: Fitted preprocessor
        top_n (int): Number of top features to return
        
    Returns:
        List[Tuple[str, float]]: List of (feature, importance) tuples
    """
    if not hasattr(model, 'feature_log_prob_'):
        return []
    
    # Get feature names from vectorizer
    if hasattr(preprocessor.vectorizer, 'get_feature_names_out'):
        feature_names = preprocessor.vectorizer.get_feature_names_out()
    elif hasattr(preprocessor.vectorizer, 'get_feature_names'):
        feature_names = preprocessor.vectorizer.get_feature_names()
    else:
        return []
    
    # For binary classification, get log probabilities for spam class
    if len(model.feature_log_prob_) > 1:
        spam_log_probs = model.feature_log_prob_[1]  # Spam class
    else:
        spam_log_probs = model.feature_log_prob_[0]
    
    # Convert log probabilities to regular probabilities (relative)
    spam_probs = np.exp(spam_log_probs)
    
    # Get top features
    top_indices = np.argsort(spam_probs)[-top_n:][::-1]
    
    top_features = []
    for idx in top_indices:
        feature_name = feature_names[idx]
        importance = spam_probs[idx]
        top_features.append((feature_name, importance))
    
    return top_features


def create_sample_emails() -> List[Tuple[str, str]]:
    """
    Create sample emails for testing.
    
    Returns:
        List[Tuple[str, str]]: List of (text, expected_label) tuples
    """
    samples = [
        (
            "WINNER!! You have won a free ticket to Bahamas! Call 12345 now to claim your prize!",
            "Spam"
        ),
        (
            "Hi, are we still meeting tomorrow at 3pm? Let me know if that works for you.",
            "Not Spam"
        ),
        (
            "URGENT! Your bank account needs verification. Click here to update your information immediately.",
            "Spam"
        ),
        (
            "Hey, just checking in. How are you doing? We should catch up soon.",
            "Not Spam"
        ),
        (
            "Congratulations! You've been selected for a free iPhone. Reply YES to claim your gift.",
            "Spam"
        ),
        (
            "Meeting agenda for Friday: 1. Project updates 2. Budget review 3. Next steps",
            "Not Spam"
        ),
        (
            "You have won $1,000,000! Call now at 1-800-SCAM to claim your money!",
            "Spam"
        ),
        (
            "The quarterly report is attached. Please review and provide feedback by EOD.",
            "Not Spam"
        )
    ]
    
    return samples


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test sample emails
    samples = create_sample_emails()
    print(f"\nCreated {len(samples)} sample emails")
    
    # Test probability formatting
    test_probs = [0.123456, 0.5, 0.987654]
    for prob in test_probs:
        formatted = format_probability(prob)
        print(f"Probability {prob:.6f} -> {formatted}")
    
    print("\nUtility functions tested successfully!")