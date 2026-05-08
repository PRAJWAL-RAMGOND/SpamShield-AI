#!/usr/bin/env python3
"""
Quick setup script for the Intelligent Email Filtering System.
Installs dependencies and downloads required NLTK data.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("=" * 70)
    print("INTELLIGENT EMAIL FILTERING SYSTEM - SETUP")
    print("=" * 70)
    
    # Step 1: Install dependencies
    print("\nStep 1: Installing Python dependencies...")
    if run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("   All dependencies installed successfully!")
    else:
        print("   Warning: Some dependencies may not have installed correctly")
    
    # Step 2: Download NLTK data
    print("\nStep 2: Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download punkt tokenizer
        print("   Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        print("   ✓ Punkt tokenizer downloaded")
        
        # Download stopwords
        print("   Downloading stopwords...")
        nltk.download('stopwords', quiet=True)
        print("   ✓ Stopwords downloaded")
        
        # Download punkt_tab (for newer NLTK versions)
        try:
            nltk.download('punkt_tab', quiet=True)
            print("   ✓ Punkt_tab downloaded")
        except:
            pass
        
        print("   All NLTK data downloaded successfully!")
        
    except Exception as e:
        print(f"   ✗ Error downloading NLTK data: {e}")
    
    # Step 3: Create necessary directories
    print("\nStep 3: Creating project directories...")
    directories = ['models', 'reports', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}/ directory ready")
    
    # Step 4: Create sample dataset
    print("\nStep 4: Creating sample dataset...")
    try:
        from data.sample_data import save_sample_dataset
        save_sample_dataset('data/sample_spam.csv')
        print("   ✓ Sample dataset created at data/sample_spam.csv")
    except Exception as e:
        print(f"   ✗ Error creating sample dataset: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETED!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("1. (Optional) Download the SMS Spam Collection dataset:")
    print("   - Visit: https://www.kaggle.com/uciml/sms-spam-collection-dataset")
    print("   - Save as: data/spam.csv")
    print()
    print("2. Train the models:")
    print("   python train_model.py")
    print()
    print("3. Launch the web interface:")
    print("   streamlit run app.py")
    print()
    print("4. Verify all requirements:")
    print("   python verify_requirements.py")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
