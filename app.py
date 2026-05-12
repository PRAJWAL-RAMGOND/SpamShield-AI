"""
Streamlit web interface for the intelligent email filtering system.
Provides real-time spam classification and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    predict_email, get_model_info, format_probability,
    create_sample_emails, load_metrics
)


# Page configuration
st.set_page_config(
    page_title="Intelligent Email Filtering System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .spam-box {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .ham-box {
        background-color: #ccffcc;
        border-left: 5px solid #00cc00;
    }
    .metric-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def load_models_data():
    """Load model data and metrics."""
    try:
        model_info = get_model_info()
        metrics = load_metrics()
        return model_info, metrics
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


def display_model_comparison(metrics):
    """Display model comparison metrics."""
    if not metrics:
        return
    
    st.markdown("### Model Performance Comparison")
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, model_metrics in metrics.items():
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': model_metrics.get('accuracy', 0),
            'Precision': model_metrics.get('precision', 0),
            'Recall': model_metrics.get('recall', 0),
            'F1-Score': model_metrics.get('f1_score', 0)
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    for idx, row in df.iterrows():
        if idx == 0:
            with col1:
                st.metric(
                    label=f"🏆 {row['Model']}",
                    value=f"{row['Accuracy']:.2%}",
                    delta="Best Accuracy"
                )
        elif idx == 1:
            with col2:
                st.metric(
                    label=f"🥈 {row['Model']}",
                    value=f"{row['Accuracy']:.2%}",
                    delta=f"{row['Accuracy'] - df.iloc[0]['Accuracy']:.2%}"
                )
        elif idx == 2:
            with col3:
                st.metric(
                    label=f"🥉 {row['Model']}",
                    value=f"{row['Accuracy']:.2%}",
                    delta=f"{row['Accuracy'] - df.iloc[0]['Accuracy']:.2%}"
                )
    
    # Show detailed comparison table
    with st.expander("Detailed Model Metrics"):
        st.dataframe(df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}'
        }))
        
        # Create visualization
        fig = px.bar(df.melt(id_vars=['Model'], 
                            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            var_name='Metric', value_name='Score'),
                    x='Model', y='Score', color='Metric',
                    barmode='group', title='Model Performance Comparison',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score',
            yaxis=dict(tickformat='.0%'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_prediction_interface():
    """Display the email prediction interface."""
    st.markdown("### Real-time Email Classification")
    
    # Model selection
    model_info, _ = load_models_data()
    if model_info and model_info['available_models']:
        selected_model = st.selectbox(
            "Select Classification Model",
            model_info['available_models'],
            format_func=lambda x: x.replace('_', ' ').title(),
            index=0
        )
    else:
        selected_model = 'multinomial_nb'
        st.warning("Using default model. Train models first for better accuracy.")
    
    # Email input
    email_text = st.text_area(
        "Enter Email Text",
        height=150,
        placeholder="Paste email content here...\n\nExample: WINNER!! You have won a free ticket to Bahamas! Call now to claim your prize."
    )
    
    # Sample emails
    samples = create_sample_emails()
    sample_options = ["Select a sample email..."] + [f"{text[:50]}..." for text, _ in samples]
    selected_sample = st.selectbox("Or choose a sample email", sample_options)
    
    if selected_sample != "Select a sample email...":
        sample_idx = sample_options.index(selected_sample) - 1
        email_text = samples[sample_idx][0]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        predict_button = st.button("🔍 Classify Email", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if predict_button and email_text:
        try:
            with st.spinner("Analyzing email..."):
                # Make prediction
                probability, label = predict_email(email_text, selected_model)
                
                # Display results
                st.markdown("### Classification Results")
                
                # Create prediction box
                box_class = "spam-box" if label == "Spam" else "ham-box"
                st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
                
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Prediction:** {label}")
                    st.markdown(f"**Confidence:** {format_probability(probability)}")
                    st.markdown(f"**Model Used:** {selected_model.replace('_', ' ').title()}")
                
                with col_b:
                    # Progress bar for probability
                    st.progress(float(probability))
                    
                    # Emoji based on prediction
                    if label == "Spam":
                        st.markdown("⚠️ **Likely Spam**")
                    else:
                        st.markdown("✅ **Likely Legitimate**")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show explanation
                with st.expander("How was this decision made?"):
                    st.markdown(f"""
                    The {selected_model.replace('_', ' ').title()} model analyzed your email and determined:
                    
                    - **Spam Probability:** {format_probability(probability)}
                    - **Threshold:** 50% (emails with ≥50% spam probability are classified as spam)
                    - **Classification:** {label}
                    
                    **Key Factors Considered:**
                    - Word frequency and patterns
                    - Presence of spam indicators (e.g., "WINNER", "FREE", "URGENT")
                    - Overall message structure
                    - Comparison with training data patterns
                    """)
                    
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Make sure models are trained first. Run: `python train_model.py`")


def display_training_status():
    """Display training status and instructions."""
    st.markdown("### System Status")
    
    model_info, metrics = load_models_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model_info and model_info['available_models']:
            st.success("✅ Models Trained")
            st.metric("Best Model", model_info['best_model'].replace('_', ' ').title())
        else:
            st.warning("⚠️ Models Not Trained")
    
    with col2:
        if metrics:
            best_accuracy = model_info.get('best_accuracy', 0) if model_info else 0
            st.metric("Best Accuracy", f"{best_accuracy:.2%}")
        else:
            st.metric("Accuracy", "N/A")
    
    with col3:
        dataset_exists = os.path.exists('data/spam.csv')
        if dataset_exists:
            st.success("✅ Dataset Available")
        else:
            st.warning("⚠️ Using Sample Data")
    
    # Training instructions
    with st.expander("Training Instructions"):
        st.markdown("""
        **To train the models:**
        
        1. **Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
        
        2. **Download NLTK data:**
        ```bash
        python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
        ```
        
        3. **Download dataset** (optional but recommended):
        - Download the SMS Spam Collection dataset from Kaggle
        - Place it as `data/spam.csv`
        
        4. **Train models:**
        ```bash
        python train_model.py
        ```
        
        5. **Launch this app:**
        ```bash
        streamlit run app.py
        ```
        """)


def display_algorithm_explanation():
    """Display explanation of Naïve Bayes algorithms."""
    st.markdown("### Algorithm Comparison")
    
    tab1, tab2, tab3 = st.tabs([
        "Multinomial Naïve Bayes",
        "Bernoulli Naïve Bayes", 
        "Gaussian Naïve Bayes"
    ])
    
    with tab1:
        st.markdown("""
        **Best for text classification with word counts**
        
        **How it works:**
        - Models the frequency of words in documents
        - Uses word counts as features
        - Assumes word occurrences are independent
        - Calculates probability based on word frequencies
        
        **Strengths:**
        - Excellent for text classification
        - Handles multiple occurrences of words
        - Works well with TF-IDF features
        
        **Use case:** Email spam filtering, document classification
        """)
    
    with tab2:
        st.markdown("""
        **Best for binary/boolean features**
        
        **How it works:**
        - Models presence/absence of words
        - Uses binary features (word present or not)
        - Assumes features are independent
        - Calculates probability based on word presence
        
        **Strengths:**
        - Good for short texts
        - Efficient with binary data
        - Works well with bag-of-words
        
        **Use case:** Short message classification, binary feature data
        """)
    
    with tab3:
        st.markdown("""
        **Best for continuous features**
        
        **How it works:**
        - Assumes features follow Gaussian distribution
        - Uses continuous feature values
        - Calculates probability using Gaussian PDF
        - Requires feature normalization
        
        **Strengths:**
        - Handles continuous data well
        - Works with normalized features
        - Good for numerical data
        
        **Use case:** Numerical feature classification, continuous data
        """)


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📧 SpamShield AI: Intelligent Email Filtering System</h1>', unsafe_allow_html=True)
    st.markdown("""
    *Automatically classify emails as Spam or Not Spam using Naïve Bayes algorithms*
    
    This system compares three Naïve Bayes classifiers and provides real-time email classification.
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Dashboard", "🔍 Classify Email", "📊 Model Comparison", "🤖 Algorithms", "📚 About"]
        )
        
        st.markdown("---")
        st.markdown("### System Info")
        
        model_info, _ = load_models_data()
        if model_info and model_info['available_models']:
            st.success(f"✅ {len(model_info['available_models'])} models available")
            st.info(f"Best: {model_info.get('best_model', 'N/A').replace('_', ' ').title()}")
        else:
            st.warning("⚠️ Train models first")
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("🔄 Retrain Models", use_container_width=True):
            st.info("Run in terminal: `python train_model.py`")
        
        if st.button("📊 View Reports", use_container_width=True):
            reports_dir = 'reports'
            if os.path.exists(reports_dir):
                st.success(f"Reports available in {reports_dir}/")
            else:
                st.warning("Train models first to generate reports")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        display_training_status()
        display_model_comparison(load_metrics())
        
        st.markdown("---")
        st.markdown("### Quick Classification")
        display_prediction_interface()
    
    elif page == "🔍 Classify Email":
        display_prediction_interface()
        
        st.markdown("---")
        st.markdown("### Recent Classifications")
        
        # Placeholder for history (would need database in real implementation)
        st.info("Classification history would be stored here in a production system.")
    
    elif page == "📊 Model Comparison":
        display_model_comparison(load_metrics())
        
        # Load and display confusion matrices if available
        reports_dir = 'reports'
        confusion_matrix_path = os.path.join(reports_dir, 'confusion_matrices.png')
        
        if os.path.exists(confusion_matrix_path):
            st.markdown("### Confusion Matrices")
            st.image(confusion_matrix_path, caption="Model Confusion Matrices", use_column_width=True)
        
        # Load and display ROC curves if available
        roc_path = os.path.join(reports_dir, 'roc_curves.html')
        if os.path.exists(roc_path):
            with open(roc_path, 'r', encoding='utf-8') as f:
                roc_html = f.read()
            st.components.v1.html(roc_html, height=500)
    
    elif page == "🤖 Algorithms":
        display_algorithm_explanation()
        
        st.markdown("---")
        st.markdown("### Bayes Theorem")
        
        st.latex(r'''
        P(Spam|Words) = \frac{P(Words|Spam) \cdot P(Spam)}{P(Words)}
        ''')
        
        st.markdown("""
        Where:
        - $P(Spam|Words)$: Probability email is spam given the words
        - $P(Words|Spam)$: Probability of seeing these words in spam emails
        - $P(Spam)$: Prior probability of an email being spam
        - $P(Words)$: Probability of seeing these words in any email
        
        **Naïve Assumption:** Words are independent given the class
        """)
    
    elif page == "📚 About":
        st.markdown("""
        ### About This Project
        
        **Intelligent Email Filtering System**
        
        This project implements an intelligent email filtering system using Naïve Bayes
        classification algorithms. It automatically classifies emails as Spam or Not Spam
        based on their content.
        
        **Key Features:**
        - Three Naïve Bayes algorithms compared
        - Real-time email classification
        - Comprehensive model evaluation
        - Interactive web interface
        - Visual performance metrics
        
        **Educational Value:**
        - Demonstrates Bayes theorem application
        - Shows text preprocessing techniques
        - Compares different classification approaches
        - Provides hands-on machine learning experience
        
        **Technologies Used:**
        - Python, Scikit-learn, Pandas, NLTK
        - Streamlit for web interface
        - Plotly for visualizations
        
        **Dataset:** SMS Spam Collection Dataset
        """)
        
        st.markdown("---")
        st.markdown("### Project Requirements")
        
        st.markdown("""
        This project was designed to meet the following requirements:
        
        1. **Implement Naïve Bayes classification** for spam detection
        2. **Compare three variants:** Multinomial, Bernoulli, and Gaussian Naïve Bayes
        3. **Preprocess email text** using tokenization, stop-word removal, and stemming
        4. **Create real-time classification interface**
        5. **Visualize performance metrics** including confusion matrices
        6. **Provide accuracy comparison dashboard**
        
        All requirements have been successfully implemented in this system.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Intelligent Email Filtering System • Built with Streamlit • "
        "Educational Machine Learning Project"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()