#!/usr/bin/env python3
"""Quick interactive spam checker."""

import sys
sys.path.insert(0, '.')

from src.utils import predict_email

def main():
    print("\n" + "="*60)
    print("SPAM EMAIL CHECKER - Interactive Mode")
    print("="*60)
    print("\nEnter 'quit' to exit\n")
    
    while True:
        # Get email from user
        print("-" * 60)
        email_text = input("\nEnter email text to check: ").strip()
        
        if email_text.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if not email_text:
            print("❌ Please enter some text!")
            continue
        
        # Check spam
        try:
            probability, label = predict_email(email_text)
            
            # Display result
            print("\n" + "="*60)
            if label == "Spam":
                print("⚠️  RESULT: SPAM")
                print(f"   Confidence: {probability*100:.2f}%")
                print("   This email is likely malicious or unwanted")
            else:
                print("✅ RESULT: NOT SPAM")
                print(f"   Confidence: {(1-probability)*100:.2f}%")
                print("   This email appears to be legitimate")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure models are trained: python train_model.py")
            break

if __name__ == "__main__":
    main()
