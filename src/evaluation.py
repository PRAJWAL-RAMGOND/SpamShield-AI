"""
Model evaluation and visualization module.
Provides comprehensive evaluation metrics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModelEvaluator:
    """Evaluates and visualizes model performance."""
    
    def __init__(self, models, metrics, X_test, y_test):
        """
        Initialize evaluator.
        
        Args:
            models (dict): Dictionary of trained models
            metrics (dict): Dictionary of model metrics
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        """
        self.models = models
        self.metrics = metrics
        self.X_test = X_test
        self.y_test = y_test
        
    def create_comparison_dataframe(self):
        """
        Create a DataFrame comparing all models.
        
        Returns:
            pandas.DataFrame: Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, metric_dict in self.metrics.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metric_dict['accuracy'],
                'Precision': metric_dict['precision'],
                'Recall': metric_dict['recall'],
                'F1-Score': metric_dict['f1_score']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        return df
    
    def plot_confusion_matrices(self, figsize=(15, 5)):
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Get predictions
            if model_name == 'bernoulli_nb':
                # Bernoulli NB requires binary features
                X_test_binary = (self.X_test > 0).astype(int)
                y_pred = model.predict(X_test_binary)
            else:
                y_pred = model.predict(self.X_test)
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Plot
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            ax.set_title(f'{model_name.replace("_", " ").title()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self):
        """
        Create a bar chart comparing model metrics.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        df = self.create_comparison_dataframe()
        
        # Melt the DataFrame for plotting
        df_melted = df.melt(id_vars=['Model'], 
                           value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                           var_name='Metric', value_name='Score')
        
        # Create bar chart
        fig = px.bar(df_melted, x='Model', y='Score', color='Metric',
                    barmode='group', title='Model Performance Comparison',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            showlegend=True
        )
        
        return fig
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for models that support probability predictions.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each model
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                # Get probabilities
                if model_name == 'bernoulli_nb':
                    X_test_binary = (self.X_test > 0).astype(int)
                    y_proba = model.predict_proba(X_test_binary)[:, 1]
                else:
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})',
                    line=dict(color=colors[idx % len(colors)], width=3)
                ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_precision_recall_curves(self):
        """
        Plot precision-recall curves.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                # Get probabilities
                if model_name == 'bernoulli_nb':
                    X_test_binary = (self.X_test > 0).astype(int)
                    y_proba = model.predict_proba(X_test_binary)[:, 1]
                else:
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{model_name.replace("_", " ").title()}',
                    line=dict(color=colors[idx % len(colors)], width=3)
                ))
        
        fig.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            dict: Report dictionary with all metrics and visualizations
        """
        report = {
            'comparison_df': self.create_comparison_dataframe(),
            'best_model': None,
            'best_accuracy': 0,
            'detailed_metrics': {}
        }
        
        # Find best model
        for model_name, metric_dict in self.metrics.items():
            report['detailed_metrics'][model_name] = metric_dict
            
            if metric_dict['accuracy'] > report['best_accuracy']:
                report['best_accuracy'] = metric_dict['accuracy']
                report['best_model'] = model_name
        
        # Generate predictions for each model
        report['predictions'] = {}
        for model_name, model in self.models.items():
            if model_name == 'bernoulli_nb':
                X_test_binary = (self.X_test > 0).astype(int)
                y_pred = model.predict(X_test_binary)
            else:
                y_pred = model.predict(self.X_test)
            
            report['predictions'][model_name] = y_pred
        
        # Generate classification reports
        report['classification_reports'] = {}
        for model_name, y_pred in report['predictions'].items():
            report['classification_reports'][model_name] = classification_report(
                self.y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True
            )
        
        return report
    
    def save_visualizations(self, directory='reports'):
        """
        Save all visualizations to files.
        
        Args:
            directory (str): Directory to save visualizations
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save comparison DataFrame
        df = self.create_comparison_dataframe()
        df.to_csv(f'{directory}/model_comparison.csv', index=False)
        
        # Save confusion matrices
        fig_cm = self.plot_confusion_matrices()
        fig_cm.savefig(f'{directory}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        # Save Plotly figures
        fig_metrics = self.plot_metrics_comparison()
        fig_metrics.write_html(f'{directory}/metrics_comparison.html')
        
        fig_roc = self.plot_roc_curves()
        fig_roc.write_html(f'{directory}/roc_curves.html')
        
        fig_pr = self.plot_precision_recall_curves()
        fig_pr.write_html(f'{directory}/precision_recall_curves.html')
        
        # Generate and save comprehensive report
        report = self.generate_comprehensive_report()
        import json
        with open(f'{directory}/evaluation_report.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            report_serializable = {}
            for key, value in report.items():
                if key == 'comparison_df':
                    report_serializable[key] = value.to_dict()
                elif key == 'detailed_metrics':
                    report_serializable[key] = {}
                    for model_name, metric_dict in value.items():
                        report_serializable[key][model_name] = {}
                        for metric_key, metric_value in metric_dict.items():
                            if metric_key == 'confusion_matrix':
                                report_serializable[key][model_name][metric_key] = metric_value.tolist()
                            else:
                                report_serializable[key][model_name][metric_key] = metric_value
                elif key == 'predictions':
                    report_serializable[key] = {}
                    for model_name, pred_array in value.items():
                        report_serializable[key][model_name] = pred_array.tolist()
                elif key == 'classification_reports':
                    report_serializable[key] = value
                else:
                    report_serializable[key] = value
            
            json.dump(report_serializable, f, indent=2)
        
        print(f"Visualizations and reports saved to {directory}/")


def evaluate_models(trainer):
    """
    Evaluate trained models and generate visualizations.
    
    Args:
        trainer (NaiveBayesTrainer): Trained model trainer
        
    Returns:
        ModelEvaluator: Evaluator instance
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        trainer.models,
        trainer.metrics,
        trainer.X_test,
        trainer.y_test
    )
    
    # Generate comparison
    print("\nModel Comparison:")
    df = evaluator.create_comparison_dataframe()
    print(df.to_string(index=False))
    
    # Get best model
    best_name, _, best_metrics = trainer.get_best_model()
    print(f"\nBest Model: {best_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Save visualizations
    print("\nGenerating visualizations...")
    evaluator.save_visualizations('reports')
    
    print("\nEvaluation completed!")
    return evaluator


if __name__ == "__main__":
    # This would typically be called after training
    print("Run this module after training models.")
    print("Example usage:")
    print("  from src.training import train_and_save_model")
    print("  from src.evaluation import evaluate_models")
    print("  ")
    print("  trainer = train_and_save_model()")
    print("  evaluator = evaluate_models(trainer)")