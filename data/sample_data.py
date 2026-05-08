"""
Sample dataset for demonstration when real dataset is not available.
"""

import pandas as pd
import numpy as np


def create_sample_dataset():
    """Create a sample spam/ham dataset for demonstration."""
    
    # Sample spam messages
    spam_messages = [
        "WINNER!! You have won a free ticket to Bahamas! Call 12345 now to claim your prize!",
        "URGENT! Your bank account needs verification. Click here to update your information immediately.",
        "Congratulations! You've been selected for a free iPhone. Reply YES to claim your gift.",
        "You have won $1,000,000! Call now at 1-800-SCAM to claim your money!",
        "Limited time offer! Get 50% off on all products. Click here to shop now!",
        "Your account has been compromised. Verify your identity now to secure your account.",
        "Exclusive offer for you! Get a free vacation package. Call now to book!",
        "Important: Your subscription is about to expire. Renew now to avoid service interruption.",
        "You've been chosen for a special reward! Claim your $500 gift card now!",
        "Warning: Unusual activity detected on your account. Secure it now!",
        "Get rich quick! Earn $5000 per week working from home. No experience needed!",
        "Your package delivery failed. Click here to reschedule and claim your package.",
        "You have unread messages from attractive singles in your area. View now!",
        "Your credit score has been updated. Check your new score for free!",
        "You're eligible for a government grant! Apply now to receive $10,000!",
        "Your friend has sent you a money transfer. Click to claim your funds!",
        "Your device may be infected. Download our antivirus software for free!",
        "You've won a luxury car! Complete the survey to claim your prize!",
        "Your social media account has suspicious activity. Secure it now!",
        "Get paid to take surveys! Earn up to $100 per day from home."
    ]
    
    # Sample ham (legitimate) messages
    ham_messages = [
        "Hi, are we still meeting tomorrow at 3pm? Let me know if that works for you.",
        "Hey, just checking in. How are you doing? We should catch up soon.",
        "Meeting agenda for Friday: 1. Project updates 2. Budget review 3. Next steps",
        "The quarterly report is attached. Please review and provide feedback by EOD.",
        "Can you send me the presentation slides from yesterday's meeting?",
        "Don't forget about the team lunch tomorrow at 12:30 at the usual place.",
        "I'll be working from home today. Let me know if you need anything.",
        "The project deadline has been extended to next Friday. Please adjust your schedules.",
        "Great work on the presentation yesterday! The client was very impressed.",
        "Reminder: Team building event this Friday at 4pm. Please RSVP by tomorrow.",
        "I've uploaded the documents to the shared drive. Let me know if you can access them.",
        "Can we schedule a quick call to discuss the budget proposal?",
        "The conference room for tomorrow's meeting has been changed to room 302.",
        "Please submit your timesheets by end of day today for payroll processing.",
        "The software update has been completed. Please restart your computers.",
        "We're ordering lunch for the team. What would you like?",
        "The client has requested some changes to the design. Let's discuss tomorrow.",
        "I'll be out of office next week. John will be covering for me.",
        "Happy birthday! Hope you have a wonderful day celebrating!",
        "The monthly newsletter has been published. Check it out when you have time."
    ]
    
    # Create DataFrame
    spam_df = pd.DataFrame({
        'label': ['spam'] * len(spam_messages),
        'text': spam_messages
    })
    
    ham_df = pd.DataFrame({
        'label': ['ham'] * len(ham_messages),
        'text': ham_messages
    })
    
    # Combine and shuffle
    df = pd.concat([spam_df, ham_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def save_sample_dataset(filepath='data/sample_spam.csv'):
    """Save sample dataset to CSV file."""
    df = create_sample_dataset()
    df.to_csv(filepath, index=False)
    print(f"Sample dataset saved to {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"Spam samples: {len(df[df['label'] == 'spam'])}")
    print(f"Ham samples: {len(df[df['label'] == 'ham'])}")
    return df


if __name__ == "__main__":
    # Create and save sample dataset
    df = save_sample_dataset()
    print("\nFirst 5 samples:")
    print(df.head())