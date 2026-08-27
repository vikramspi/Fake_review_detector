"""
📊 TEST & DEMO SCRIPT FOR FAKE REVIEW DETECTOR
Run this script to test the detector without Streamlit UI
"""

import sys
sys.path.append('.')

from fake_review_detector import (
    extract_features, 
    load_or_train_model, 
    predict_review, 
    get_risk_factors
)
import pandas as pd

# Sample test reviews
TEST_REVIEWS = {
    "likely_fake_short": "Great product! Really good. Very nice. Awesome quality. Love it. 5 stars.",
    
    "likely_fake_repetitive": "Amazing amazing amazing! So good so good! Really really really great. Best best best product!",
    
    "likely_genuine_detailed": """This product exceeded my expectations in terms of build quality and durability. 
    The materials used are premium and the attention to detail is evident throughout. 
    However, I found the instruction manual a bit confusing at first. Once I figured it out, 
    the setup process was straightforward. Would definitely recommend to anyone looking 
    for a reliable solution in this category. The customer service was also responsive and helpful.""",
    
    "likely_genuine_balanced": """Pros: Excellent build quality, fast shipping, and responsive customer service. 
    The product works exactly as described on the listing. 
    Cons: The packaging could have been more secure, and it arrived with a small dent in one corner, 
    though it doesn't affect functionality. Overall, very satisfied with this purchase.""",
    
    "likely_fake_structured": "Five star product. Great quality. Highly recommended. Best choice. Perfect item.",
    
    "medium_length_review": "This product works well for the price. It's not the best I've used, but it's definitely reliable. Would buy again if needed."
}

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def test_detector():
    """Test the detector with sample reviews"""
    
    print_header("🔍 FAKE REVIEW DETECTOR - TEST SUITE")
    
    # Load model
    print("📦 Loading/Training model...")
    model = load_or_train_model()
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!\n")
    
    # Test each review
    for review_name, review_text in TEST_REVIEWS.items():
        print_header(f"Testing: {review_name.upper()}")
        print(f"Review: {review_text[:100]}..." if len(review_text) > 100 else f"Review: {review_text}\n")
        
        # Get prediction
        prediction, confidence, features = predict_review(review_text, model)
        
        if prediction is not None:
            # Display results
            label = "🤖 FAKE/CG" if prediction == 1 else "✅ GENUINE/OR"
            fake_prob = confidence[1] * 100
            real_prob = confidence[0] * 100
            
            print(f"Prediction: {label}")
            print(f"Fake Probability: {fake_prob:.1f}%")
            print(f"Real Probability: {real_prob:.1f}%")
            
            # Feature details
            print(f"\n📊 Feature Analysis:")
            print(f"  • Text Length: {features['text_length']} characters")
            print(f"  • Vocabulary Richness: {features['vocab_richness']:.3f}")
            print(f"  • Avg Word Length: {features['avg_word_len']:.2f}")
            print(f"  • Sentence Count: {features['sentence_count']:.0f}")
            print(f"  • Punctuation Ratio: {features['punctuation_ratio']:.2%}")
            print(f"  • Uppercase Ratio: {features['uppercase_ratio']:.2%}")
            print(f"  • Special Char Ratio: {features['special_char_ratio']:.2%}")
            print(f"  • Repetition Score: {features['repetition_score']:.3f}")
            print(f"  • Avg Sentence Length: {features['avg_sentence_length']:.1f} words")
            
            # Risk factors
            risk_factors = get_risk_factors(features, confidence)
            if risk_factors:
                print(f"\n⚠️ Risk Factors:")
                for factor in risk_factors:
                    print(f"  {factor}")
            else:
                print(f"\n✅ No major risk factors detected")
        else:
            print("❌ Prediction failed")

def interactive_mode():
    """Interactive testing mode"""
    print_header("🔍 FAKE REVIEW DETECTOR - INTERACTIVE MODE")
    
    # Load model
    print("📦 Loading/Training model...")
    model = load_or_train_model()
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    print("Type 'quit' to exit\n")
    
    while True:
        print("-" * 80)
        review = input("📝 Enter review text (or 'quit' to exit):\n> ")
        
        if review.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not review.strip():
            print("⚠️ Please enter a review")
            continue
        
        # Get prediction
        prediction, confidence, features = predict_review(review, model)
        
        if prediction is not None:
            label = "🤖 FAKE/CG" if prediction == 1 else "✅ GENUINE/OR"
            fake_prob = confidence[1] * 100
            real_prob = confidence[0] * 100
            
            print(f"\n📊 Result: {label}")
            print(f"Fake Probability: {fake_prob:.1f}%")
            print(f"Real Probability: {real_prob:.1f}%")
            
            risk_factors = get_risk_factors(features, confidence)
            if risk_factors:
                print(f"⚠️ Risk Factors:")
                for factor in risk_factors:
                    print(f"  {factor}")
            else:
                print(f"✅ No major risk factors detected")
        else:
            print("❌ Prediction failed")

def batch_test_csv():
    """Test detector on entire CSV dataset"""
    import os
    
    csv_path = 'Final_enhanced_dataset (1).csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print_header("📊 BATCH TESTING CSV DATASET")
    
    # Load model
    print("📦 Loading/Training model...")
    model = load_or_train_model()
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!\n")
    
    # Load CSV
    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} reviews\n")
    
    # Test first 10 of each type
    print("Testing first 5 CG (fake) and 5 OR (genuine) reviews:\n")
    
    for label_type in ['CG', 'OR']:
        subset = df[df['label'] == label_type].head(5)
        label_display = "🤖 FAKE/CG" if label_type == 'CG' else "✅ GENUINE/OR"
        
        print(f"\n{label_display} Reviews:")
        print("-" * 80)
        
        for idx, row in subset.iterrows():
            prediction, confidence, _ = predict_review(row['text_'], model)
            
            if prediction is not None:
                pred_label = "🤖 FAKE" if prediction == 1 else "✅ REAL"
                fake_prob = confidence[1] * 100
                
                print(f"Expected: {label_display} | Predicted: {pred_label} | Confidence: {fake_prob:.1f}%")
            else:
                print(f"❌ Failed to predict")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fake Review Detector Test Suite")
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'interactive', 'batch'],
                       help='Test mode: test (sample reviews), interactive (user input), batch (CSV)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_detector()
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'batch':
        batch_test_csv()
    
    print("\n✅ Testing complete!")
