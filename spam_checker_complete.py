#!/usr/bin/env python3
"""
Complete spam checker with all features.
"""

import sys
sys.path.insert(0, '.')

from src.utils import predict_email, create_sample_emails

def check_email(email_text, model='multinomial_nb', show_details=True):
    """
    Check if an email is spam.
    
    Args:
        email_text (str): Email content to check
        model (str): Model to use (multinomial_nb, bernoulli_nb, gaussian_nb)
        show_details (bool): Show detailed output
    
    Returns:
        tuple: (probability, label)
    """
    probability, label = predict_email(email_text, model)
    
    if show_details:
        print("\n" + "="*70)
        print("SPAM CLASSIFICATION RESULT")
        print("="*70)
        
        # Email preview
        preview = email_text[:100] + "..." if len(email_text) > 100 else email_text
        print(f"\n📧 Email: {preview}")
        
        # Classification
        if label == "Spam":
            print(f"\n⚠️  Classification: SPAM")
            print(f"   Spam Probability: {probability*100:.2f}%")
            print(f"   Recommendation: DELETE or MARK AS SPAM")
        else:
            print(f"\n✅ Classification: NOT SPAM (Legitimate)")
            print(f"   Ham Probability: {(1-probability)*100:.2f}%")
            print(f"   Recommendation: SAFE TO READ")
        
        # Confidence
        confidence = abs(probability - 0.5) * 2
        print(f"\n📊 Confidence Level: {confidence*100:.1f}%")
        
        if confidence > 0.8:
            print("   Very confident in this classification")
        elif confidence > 0.5:
            print("   Moderately confident in this classification")
        else:
            print("   Low confidence - manual review recommended")
        
        # Model info
        print(f"\n🤖 Model: {model.replace('_', ' ').title()}")
        
        print("="*70 + "\n")
    
    return probability, label


def test_sample_emails():
    """Test with sample emails."""
    print("\n" + "="*70)
    print("TESTING WITH SAMPLE EMAILS")
    print("="*70)
    
    samples = create_sample_emails()
    
    for i, (text, expected) in enumerate(samples, 1):
        print(f"\n--- Sample {i}/{len(samples)} ---")
        print(f"Expected: {expected}")
        
        probability, label = check_email(text, show_details=False)
        
        # Short output
        status = "✅ CORRECT" if label == expected else "❌ WRONG"
        print(f"Result: {label} ({probability*100:.1f}%) {status}")
        
        if i < len(samples):
            input("\nPress Enter for next sample...")


def interactive_mode():
    """Interactive spam checking."""
    print("\n" + "="*70)
    print("INTERACTIVE SPAM CHECKER")
    print("="*70)
    print("\nCommands:")
    print("  - Type email text to check")
    print("  - Type 'samples' to test sample emails")
    print("  - Type 'quit' to exit")
    print("="*70)
    
    while True:
        print("\n" + "-"*70)
        user_input = input("\nEnter email text (or command): ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        elif user_input.lower() == 'samples':
            test_sample_emails()
        
        elif user_input:
            check_email(user_input)
        
        else:
            print("❌ Please enter some text!")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("INTELLIGENT EMAIL SPAM CHECKER")
    print("="*70)
    print("\nOptions:")
    print("1. Check a specific email")
    print("2. Test sample emails")
    print("3. Interactive mode")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        email = input("\nEnter email text: ").strip()
        if email:
            check_email(email)
        else:
            print("❌ No text entered!")
    
    elif choice == '2':
        test_sample_emails()
    
    elif choice == '3':
        interactive_mode()
    
    elif choice == '4':
        print("\n👋 Goodbye!")
    
    else:
        print("❌ Invalid option!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure models are trained: python train_model.py")
