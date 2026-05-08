"""
Model training module for email filtering system.
Trains and compares multiple Naïve Bayes classifiers.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

from src.preprocessing import EmailPreprocessor, load_sms_dataset


class NaiveBayesTrainer:
    """Trains and compares multiple Naïve Bayes classifiers."""
    
    def __init__(self, preprocessor=None):
        """
        Initialize the trainer.
        
        Args:
            preprocessor (EmailPreprocessor): Preprocessor instance
        """
        self.preprocessor = preprocessor or EmailPreprocessor()
        self.models = {}
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Prepare and split the dataset.
        
        Args:
            texts (list): List of email texts
            labels (list): List of labels (0=ham, 1=spam)
            test_size (float): Proportion of test data
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Preprocess texts
        print("Preprocessing texts...")
        processed_texts = self.preprocessor.preprocess_dataset(texts)
        
        # Create feature vectors
        print("Creating feature vectors...")
        X = self.preprocessor.fit_transform(processed_texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_multinomial_nb(self, alpha=1.0):
        """
        Train Multinomial Naïve Bayes classifier.
        
        Args:
            alpha (float): Smoothing parameter
            
        Returns:
            MultinomialNB: Trained model
        """
        print("\nTraining Multinomial Naïve Bayes...")
        model = MultinomialNB(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        self._evaluate_model(model, y_pred, "multinomial_nb")
        
        self.models["multinomial_nb"] = model
        return model
    
    def train_bernoulli_nb(self, alpha=1.0, binarize=0.0):
        """
        Train Bernoulli Naïve Bayes classifier.
        
        Args:
            alpha (float): Smoothing parameter
            binarize (float): Threshold for binarization
            
        Returns:
            BernoulliNB: Trained model
        """
        print("\nTraining Bernoulli Naïve Bayes...")
        
        # Bernoulli NB requires binary features
        X_train_binary = (self.X_train > binarize).astype(int)
        X_test_binary = (self.X_test > binarize).astype(int)
        
        model = BernoulliNB(alpha=alpha, binarize=binarize)
        model.fit(X_train_binary, self.y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_binary)
        self._evaluate_model(model, y_pred, "bernoulli_nb")
        
        self.models["bernoulli_nb"] = model
        return model
    
    def train_gaussian_nb(self):
        """
        Train Gaussian Naïve Bayes classifier.
        
        Returns:
            GaussianNB: Trained model
        """
        print("\nTraining Gaussian Naïve Bayes...")
        
        # Gaussian NB works better with normalized continuous features
        # We'll use TF-IDF values which are already continuous
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        self._evaluate_model(model, y_pred, "gaussian_nb")
        
        self.models["gaussian_nb"] = model
        return model
    
    def train_all_models(self):
        """Train all three Naïve Bayes models."""
        self.train_multinomial_nb()
        self.train_bernoulli_nb()
        self.train_gaussian_nb()
        
        return self.models
    
    def _evaluate_model(self, model, y_pred, model_name):
        """
        Evaluate a model and store metrics.
        
        Args:
            model: Trained model
            y_pred: Model predictions
            model_name (str): Name of the model
        """
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # Store metrics
        self.metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Ham', 'Spam']))
    
    def get_best_model(self):
        """
        Get the best performing model based on accuracy.
        
        Returns:
            tuple: (model_name, model, metrics)
        """
        if not self.metrics:
            return None, None, None
        
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['accuracy'])
        best_model = self.models.get(best_model_name)
        best_metrics = self.metrics.get(best_model_name)
        
        return best_model_name, best_model, best_metrics
    
    def save_models(self, directory='models'):
        """
        Save trained models to disk.
        
        Args:
            directory (str): Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = os.path.join(directory, f'{model_name}_model.pkl')
            joblib.dump(model, filename)
            print(f"Saved {model_name} model to {filename}")
        
        # Save preprocessor
        preprocessor_file = os.path.join(directory, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_file)
        print(f"Saved preprocessor to {preprocessor_file}")
        
        # Save metrics
        metrics_file = os.path.join(directory, 'metrics.json')
        import json
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_serializable = {}
            for model_name, metric_dict in self.metrics.items():
                metrics_serializable[model_name] = {}
                for key, value in metric_dict.items():
                    if key == 'confusion_matrix':
                        metrics_serializable[model_name][key] = value.tolist()
                    else:
                        metrics_serializable[model_name][key] = value
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Saved metrics to {metrics_file}")
    
    def load_models(self, directory='models'):
        """
        Load trained models from disk.
        
        Args:
            directory (str): Directory containing models
        """
        model_files = {
            'multinomial_nb': 'multinomial_nb_model.pkl',
            'bernoulli_nb': 'bernoulli_nb_model.pkl',
            'gaussian_nb': 'gaussian_nb_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name} model from {filepath}")
        
        # Load preprocessor
        preprocessor_file = os.path.join(directory, 'preprocessor.pkl')
        if os.path.exists(preprocessor_file):
            self.preprocessor = joblib.load(preprocessor_file)
            print(f"Loaded preprocessor from {preprocessor_file}")
        
        # Load metrics
        metrics_file = os.path.join(directory, 'metrics.json')
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)
            print(f"Loaded metrics from {metrics_file}")


def train_and_save_model(dataset_path='data/spam.csv'):
    """
    Complete training pipeline.
    
    Args:
        dataset_path (str): Path to dataset file
    """
    print("=" * 60)
    print("EMAIL FILTERING SYSTEM - MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    texts, labels = load_sms_dataset(dataset_path)
    print(f"   Loaded {len(texts)} samples")
    
    # Initialize trainer
    trainer = NaiveBayesTrainer()
    
    # Prepare data
    print("\n2. Preparing data...")
    trainer.prepare_data(texts, labels)
    
    # Train models
    print("\n3. Training models...")
    trainer.train_all_models()
    
    # Get best model
    best_name, best_model, best_metrics = trainer.get_best_model()
    if best_model:
        print(f"\n4. Best model: {best_name}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Save models
    print("\n5. Saving models...")
    trainer.save_models()
    
    print("\nTraining completed successfully!")
    return trainer


if __name__ == "__main__":
    # Run training pipeline
    trainer = train_and_save_model()