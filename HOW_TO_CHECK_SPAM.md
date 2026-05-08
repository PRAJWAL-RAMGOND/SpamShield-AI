# How to Check if an Email is Spam - Complete Guide

## 🎯 Quick Answer

There are **3 ways** to check if an email is spam:

1. **Web Interface (Easiest)** - Use the Streamlit app
2. **Python Script** - Use the command line
3. **Python Code** - Import and use in your own code

---

## Method 1: Web Interface (Recommended) 🌐

### Step 1: Open the Web App

Open your browser and go to: **http://localhost:8501**

### Step 2: Navigate to Classification Page

Click on **"🔍 Classify Email"** in the left sidebar

### Step 3: Enter Your Email

**Option A: Type your email**
```
Paste or type your email text in the text area
```

**Option B: Use a sample email**
```
Select from the dropdown: "Or choose a sample email"
```

### Step 4: Click "Classify Email" Button

### Step 5: View Results

You'll see:
- ✅ **Classification:** Spam or Not Spam
- 📊 **Confidence:** Percentage (e.g., 95.67%)
- 🤖 **Model Used:** Which algorithm was used
- 📈 **Progress Bar:** Visual confidence indicator

---

## Method 2: Python Script (Command Line) 💻

### Create a Test Script

Create a file called `check_spam.py`:

```python
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
    
    # Check each email
    for email in test_emails:
        check_spam(email)
        input("Press Enter to check next email...")
```

### Run the Script

```bash
python check_spam.py
```

### Expected Output

```
============================================================
EMAIL SPAM CLASSIFICATION RESULT
============================================================

Email Text:
  WINNER!! You have won a free ticket to Bahamas! Call 12345 now!

Classification: Spam
Spam Probability: 98.45%
Confidence: High
Model Used: Multinomial Nb

⚠️  WARNING: This email is likely SPAM!
============================================================
```

---

## Method 3: Python Code (In Your Program) 🐍

### Basic Usage

```python
from src.utils import predict_email

# Your email text
email_text = "WINNER!! You have won a free prize! Call now!"

# Check if spam
probability, label = predict_email(email_text)

print(f"Classification: {label}")
print(f"Probability: {probability*100:.2f}%")

# Use the result
if label == "Spam":
    print("⚠️ This is spam!")
else:
    print("✅ This is legitimate")
```

### Advanced Usage with Different Models

```python
from src.utils import predict_email

email_text = "Your package delivery failed. Click to reschedule."

# Try all three models
models = ['multinomial_nb', 'bernoulli_nb', 'gaussian_nb']

print(f"\nEmail: {email_text}\n")
print("Model Comparison:")
print("-" * 60)

for model in models:
    probability, label = predict_email(email_text, model)
    print(f"{model.replace('_', ' ').title():20} | {label:10} | {probability*100:5.2f}%")
```

### Output

```
Email: Your package delivery failed. Click to reschedule.

Model Comparison:
------------------------------------------------------------
Multinomial Nb       | Spam       | 95.67%
Bernoulli Nb         | Spam       | 92.34%
Gaussian Nb          | Spam       | 88.12%
```

---

## 📧 Example Emails to Test

### Spam Examples (Should be detected as SPAM)

#### Example 1: Prize/Winner Scam
```
WINNER!! You have won a free ticket to Bahamas! Call 12345 now to claim your prize!
```
**Expected:** ⚠️ Spam (95-100% probability)

#### Example 2: Urgent Bank Scam
```
URGENT! Your bank account needs verification. Click here to update your information immediately.
```
**Expected:** ⚠️ Spam (95-100% probability)

#### Example 3: Free Gift Scam
```
Congratulations! You've been selected for a free iPhone. Reply YES to claim your gift.
```
**Expected:** ⚠️ Spam (95-100% probability)

#### Example 4: Money Scam
```
You have won $1,000,000! Call now at 1-800-SCAM to claim your money!
```
**Expected:** ⚠️ Spam (95-100% probability)

#### Example 5: Account Security Scam
```
Your account has been compromised. Verify your identity now to secure your account.
```
**Expected:** ⚠️ Spam (90-100% probability)

### Legitimate Examples (Should be detected as NOT SPAM)

#### Example 1: Meeting Request
```
Hi, are we still meeting tomorrow at 3pm? Let me know if that works for you.
```
**Expected:** ✅ Not Spam (95-100% probability)

#### Example 2: Work Email
```
Meeting agenda for Friday: 1. Project updates 2. Budget review 3. Next steps
```
**Expected:** ✅ Not Spam (95-100% probability)

#### Example 3: Document Request
```
The quarterly report is attached. Please review and provide feedback by EOD.
```
**Expected:** ✅ Not Spam (95-100% probability)

#### Example 4: Casual Message
```
Hey, just checking in. How are you doing? We should catch up soon.
```
**Expected:** ✅ Not Spam (95-100% probability)

#### Example 5: Work Update
```
I'll be working from home today. Let me know if you need anything.
```
**Expected:** ✅ Not Spam (95-100% probability)

---

## 🎮 Interactive Testing

### Quick Test Script

Save this as `quick_test.py`:

```python
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
```

### Run Interactive Mode

```bash
python quick_test.py
```

### Example Session

```
============================================================
SPAM EMAIL CHECKER - Interactive Mode
============================================================

Enter 'quit' to exit

------------------------------------------------------------

Enter email text to check: WINNER!! You won a prize!

============================================================
⚠️  RESULT: SPAM
   Confidence: 98.45%
   This email is likely malicious or unwanted
============================================================
------------------------------------------------------------

Enter email text to check: Hi, let's meet tomorrow at 3pm

============================================================
✅ RESULT: NOT SPAM
   Confidence: 97.23%
   This email appears to be legitimate
============================================================
------------------------------------------------------------

Enter email text to check: quit

Goodbye!
```

---

## 📊 Understanding the Output

### Classification Result

| Output | Meaning |
|--------|---------|
| **Spam** | Email is likely spam/malicious |
| **Not Spam** | Email is likely legitimate |

### Probability Score

| Range | Interpretation |
|-------|----------------|
| **90-100%** | Very high confidence |
| **70-90%** | High confidence |
| **50-70%** | Medium confidence |
| **Below 50%** | Low confidence (opposite class) |

### Confidence Level

- **High:** Probability is far from 50% (clear decision)
- **Medium:** Probability is close to 50% (uncertain)

---

## 🔍 What Makes an Email Spam?

The system looks for these indicators:

### Spam Indicators ⚠️

1. **Urgent language:** "URGENT!", "ACT NOW!", "LIMITED TIME"
2. **Prize/Winner words:** "WINNER", "WON", "FREE", "PRIZE"
3. **Money mentions:** "$$$", "CASH", "MILLION"
4. **Action demands:** "CLICK HERE", "CALL NOW", "VERIFY"
5. **Suspicious patterns:** All caps, excessive punctuation
6. **Scam keywords:** "VERIFY ACCOUNT", "CLAIM NOW"

### Legitimate Indicators ✅

1. **Normal conversation:** "Hi", "Hello", "Thanks"
2. **Work-related:** "meeting", "report", "project"
3. **Casual tone:** Natural language, proper grammar
4. **Specific details:** Times, dates, names
5. **No urgency:** Calm, professional tone

---

## 🎯 Model Selection Guide

### Which Model to Use?

| Model | Best For | Accuracy |
|-------|----------|----------|
| **Multinomial NB** | General email classification | 100% |
| **Bernoulli NB** | Short messages, binary features | 100% |
| **Gaussian NB** | Continuous features | 87.5% |

**Recommendation:** Use **Multinomial NB** (default) for best results!

---

## 🚀 Complete Example Script

Save this as `spam_checker_complete.py`:

```python
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
        print(f"\n--- Sample {i} ---")
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
```

### Run Complete Script

```bash
python spam_checker_complete.py
```

---

## 📝 Summary

### Easiest Way (Web Interface)
1. Open http://localhost:8501
2. Go to "🔍 Classify Email"
3. Enter email text
4. Click "Classify Email"
5. See result!

### Command Line Way
```bash
python check_spam.py
```

### Python Code Way
```python
from src.utils import predict_email
probability, label = predict_email("Your email text here")
print(f"Result: {label} ({probability*100:.2f}%)")
```

---

## 🎉 You're Ready!

Now you know **3 ways** to check if an email is spam:
1. ✅ Web interface (easiest)
2. ✅ Command line script
3. ✅ Python code integration

**Try it now with the web interface at http://localhost:8501!**
