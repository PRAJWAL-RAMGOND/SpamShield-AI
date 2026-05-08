#!/usr/bin/env python3
"""
Quick script to check if an email is spam.
"""

import sys
sys.path.insert(0, '.')

from src.utils import predict_email

def check_spam(email_text, model_name='multinomial_nb'):
    """Check if an email is spam."""
    try:
        probability, label = predict_email(email_text, model_name)
        
        print("\n" + "="*60)
        print("EMAIL SPAM CLASSIFICATION RESULT")
        print("="*60)
        print(f"\nEmail Text:")
        print(f"  {email_text[:100]}{'...' if len(email_text) > 100 else ''}")
        print(f"\nClassification: {label}")
        print(f"Spam Probability: {probability*100:.2f}%")
        print(f"Confidence: {'High' if abs(probability - 0.5) > 0.3 else 'Medium'}")
        print(f"Model Used: {model_name.replace('_', ' ').title()}")
        
        if label == "Spam":
            print("\n⚠️  WARNING: This email is likely SPAM!")
        else:
            print("\n✅ This email appears to be LEGITIMATE")
        
        print("="*60 + "\n")
        
        return probability, label
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure models are trained. Run: python train_model.py")
        return None, None

if __name__ == "__main__":
    # Example emails to test
    test_emails = [
        "WINNER!! You have won a free ticket to Bahamas! Call 12345 now!",
        "Hi, are we still meeting tomorrow at 3pm?",
        "URGENT! Your bank account needs verification. Click here now!",
        "The quarterly report is attached. Please review by EOD."
    ]
    
    print("\n" + "="*60)
    print("SPAM EMAIL CHECKER - Testing Sample Emails")
    print("="*60)
    
    # Check each email
    for i, email in enumerate(test_emails, 1):
        print(f"\n--- Testing Email {i}/{len(test_emails)} ---")
        check_spam(email)
        
        if i < len(test_emails):
            input("Press Enter to check next email...")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
