"""
⚡ OPTIMIZED FAKE REVIEW DETECTION SYSTEM
High-performance detection with Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. FEATURE EXTRACTION FUNCTIONS (Optimized)
# ============================================================================

def extract_text_length(text):
    """Extract text length feature"""
    return len(str(text).strip())

def extract_vocabulary_richness(text):
    """Extract vocabulary richness (unique words / total words)"""
    if not text or not isinstance(text, str):
        return 0.0
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

def extract_avg_word_length(text):
    """Extract average word length"""
    if not text or not isinstance(text, str):
        return 0.0
    words = str(text).split()
    if len(words) == 0:
        return 0.0
    return np.mean([len(word) for word in words])

def extract_sentence_count(text):
    """Extract sentence count"""
    if not text or not isinstance(text, str):
        return 1
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return max(1, len(sentences))

def extract_punctuation_ratio(text):
    """Extract punctuation density"""
    if not text or len(text) == 0:
        return 0.0
    punctuation_count = sum(1 for c in text if c in '!?.,;:')
    return punctuation_count / len(text)

def extract_uppercase_ratio(text):
    """Extract uppercase character ratio"""
    if not text or len(text) == 0:
        return 0.0
    uppercase_count = sum(1 for c in text if c.isupper())
    return uppercase_count / len(text)

def extract_special_char_ratio(text):
    """Extract special character ratio"""
    if not text or len(text) == 0:
        return 0.0
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special_chars / len(text)

def extract_repetition_score(text):
    """Extract word repetition score"""
    if not text or not isinstance(text, str):
        return 0.0
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    # Calculate repetition score
    repeated_words = sum(1 for count in word_freq.values() if count > 2)
    return repeated_words / len(word_freq) if word_freq else 0.0

def extract_avg_sentence_length(text):
    """Extract average sentence length"""
    if not text or not isinstance(text, str):
        return 0.0
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0.0
    words_per_sentence = [len(s.split()) for s in sentences]
    return np.mean(words_per_sentence)

def extract_features(review_text):
    """Extract all features from review text"""
    features = {
        'text_length': extract_text_length(review_text),
        'vocab_richness': extract_vocabulary_richness(review_text),
        'avg_word_len': extract_avg_word_length(review_text),
        'sentence_count': extract_sentence_count(review_text),
        'punctuation_ratio': extract_punctuation_ratio(review_text),
        'uppercase_ratio': extract_uppercase_ratio(review_text),
        'special_char_ratio': extract_special_char_ratio(review_text),
        'repetition_score': extract_repetition_score(review_text),
        'avg_sentence_length': extract_avg_sentence_length(review_text),
    }
    return features

# ============================================================================
# 2. MODEL TRAINING (Lazy loading and caching)
# ============================================================================

@st.cache_resource
def load_or_train_model():
    """Load trained model or train a new one"""
    model_path = 'fake_review_model.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    # If no model exists, train on the provided dataset
    csv_path = 'Final_enhanced_dataset (1).csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Extract features for training
            print("🔄 Training model...")
            feature_list = []
            labels = []
            
            for idx, row in df.iterrows():
                features = extract_features(row['text_'])
                feature_list.append(list(features.values()))
                # CG = 1 (fake), OR = 0 (real)
                labels.append(1 if row['label'] == 'CG' else 0)
            
            X = np.array(feature_list)
            y = np.array(labels)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return model
        except Exception as e:
            st.warning(f"Could not load dataset for training: {e}")
            return None
    
    return None

# ============================================================================
# 3. PREDICTION AND ANALYSIS
# ============================================================================

def predict_review(review_text, model):
    """Predict if review is fake with confidence"""
    if not model:
        return None, None, None
    
    features = extract_features(review_text)
    feature_array = np.array(list(features.values())).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(feature_array)[0]
    confidence = model.predict_proba(feature_array)[0]
    
    return prediction, confidence, features

def get_risk_factors(features, confidence):
    """Identify risk factors for fake review"""
    risk_factors = []
    
    # Based on analysis from notebook
    if features['text_length'] < 100:
        risk_factors.append("⚠️ Very short review (typical of CG)")
    
    if features['vocab_richness'] < 0.7:
        risk_factors.append("⚠️ Low vocabulary richness (possible repetition)")
    
    if features['avg_word_len'] < 4.5:
        risk_factors.append("⚠️ Short average word length")
    
    if features['sentence_count'] < 2:
        risk_factors.append("⚠️ Very few sentences")
    
    if features['punctuation_ratio'] > 0.15:
        risk_factors.append("⚠️ High punctuation density")
    
    if features['uppercase_ratio'] > 0.1:
        risk_factors.append("⚠️ Excessive uppercase letters")
    
    if features['repetition_score'] > 0.3:
        risk_factors.append("⚠️ High word repetition")
    
    if features['avg_sentence_length'] < 8:
        risk_factors.append("⚠️ Very short sentences")
    
    return risk_factors

# ============================================================================
# 4. STREAMLIT UI
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="🔍 Fake Review Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom styling
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stCard {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔍 AI-POWERED FAKE REVIEW DETECTOR")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("### Detection Model Info")
        st.info("""
        **Model**: Random Forest Classifier
        - **Features**: 9 linguistic & textual features
        - **Training Data**: 44,194 reviews
        - **Accuracy**: Trained on real vs. AI-generated reviews
        """)
        
        st.markdown("### How it Works")
        st.markdown("""
        The detector analyzes:
        - 📏 Text length patterns
        - 📝 Vocabulary richness
        - 🔤 Word characteristics
        - 🔗 Sentence structure
        - 💬 Punctuation patterns
        - 🔄 Repetition indicators
        """)
    
    # Load model
    model = load_or_train_model()
    
    if model is None:
        st.error("❌ Could not load or train model. Please ensure the CSV file exists.")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔍 Detect Review", "📊 Analysis", "❓ About"])
    
    # ========== TAB 1: DETECTION ==========
    with tab1:
        st.header("Analyze Your Review")
        st.markdown("Paste a review below to detect if it's genuine or AI-generated")
        
        # Input section
        col1, col2 = st.columns([3, 1])
        with col1:
            review_input = st.text_area(
                "📝 Enter Review Text:",
                height=200,
                placeholder="Paste the review you want to analyze...",
                key="review_input"
            )
        
        with col2:
            st.markdown("### Quick Actions")
            analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and review_input.strip():
            # Get prediction
            prediction, confidence, features = predict_review(review_input, model)
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("🎯 Detection Results")
                
                # Result cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🤖 AI-GENERATED")
                        st.metric("Label", "FAKE/CG")
                    else:
                        st.success("✅ GENUINE")
                        st.metric("Label", "REAL/OR")
                
                with col2:
                    fake_confidence = confidence[1] * 100
                    st.metric("Fake Probability", f"{fake_confidence:.1f}%")
                
                with col3:
                    real_confidence = confidence[0] * 100
                    st.metric("Real Probability", f"{real_confidence:.1f}%")
                
                # Confidence visualization
                st.markdown("### Confidence Score")
                col1, col2 = st.columns(2)
                with col1:
                    st.progress(real_confidence / 100, text=f"Genuine: {real_confidence:.1f}%")
                with col2:
                    st.progress(fake_confidence / 100, text=f"AI-Generated: {fake_confidence:.1f}%")
                
                # Risk assessment
                risk_factors = get_risk_factors(features, confidence)
                
                if risk_factors:
                    st.warning("### ⚠️ Risk Factors Detected")
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.success("### ✅ No Major Risk Factors Detected")
                
                # Detailed features breakdown
                st.markdown("---")
                st.subheader("📊 Linguistic Features")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Text Length", f"{features['text_length']} chars")
                    st.metric("Vocabulary Richness", f"{features['vocab_richness']:.3f}")
                    st.metric("Avg Word Length", f"{features['avg_word_len']:.2f}")
                
                with col2:
                    st.metric("Sentence Count", f"{features['sentence_count']:.0f}")
                    st.metric("Avg Sentence Length", f"{features['avg_sentence_length']:.1f} words")
                    st.metric("Punctuation Ratio", f"{features['punctuation_ratio']:.2%}")
                
                with col3:
                    st.metric("Uppercase Ratio", f"{features['uppercase_ratio']:.2%}")
                    st.metric("Special Char Ratio", f"{features['special_char_ratio']:.2%}")
                    st.metric("Repetition Score", f"{features['repetition_score']:.3f}")
        
        elif analyze_button:
            st.warning("⚠️ Please enter a review text to analyze")
    
    # ========== TAB 2: ANALYSIS ==========
    with tab2:
        st.header("📊 Feature Analysis")
        
        if os.path.exists('Final_enhanced_dataset (1).csv'):
            df = pd.read_csv('Final_enhanced_dataset (1).csv')
            
            # Load or calculate statistics
            cg_reviews = df[df['label'] == 'CG']
            or_reviews = df[df['label'] == 'OR']
            
            st.subheader("Dataset Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Reviews", len(df))
            with col2:
                st.metric("AI-Generated (CG)", len(cg_reviews))
            with col3:
                st.metric("Genuine (OR)", len(or_reviews))
            
            # Feature comparison
            st.subheader("Feature Comparison: CG vs OR")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("### AI-Generated Reviews (CG) Stats")
                st.write(f"**Avg Text Length**: {cg_reviews['text_length'].mean():.0f} ± {cg_reviews['text_length'].std():.0f} chars")
                st.write(f"**Vocab Richness**: {cg_reviews['vocab_richness'].mean():.3f} ± {cg_reviews['vocab_richness'].std():.3f}")
                st.write(f"**Avg Word Length**: {cg_reviews['avg_word_len'].mean():.2f} ± {cg_reviews['avg_word_len'].std():.2f}")
                st.write(f"**Sentence Count**: {cg_reviews['sentence_count'].mean():.1f} ± {cg_reviews['sentence_count'].std():.1f}")
            
            with col2:
                st.success("### Genuine Reviews (OR) Stats")
                st.write(f"**Avg Text Length**: {or_reviews['text_length'].mean():.0f} ± {or_reviews['text_length'].std():.0f} chars")
                st.write(f"**Vocab Richness**: {or_reviews['vocab_richness'].mean():.3f} ± {or_reviews['vocab_richness'].std():.3f}")
                st.write(f"**Avg Word Length**: {or_reviews['avg_word_len'].mean():.2f} ± {or_reviews['avg_word_len'].std():.2f}")
                st.write(f"**Sentence Count**: {or_reviews['sentence_count'].mean():.1f} ± {or_reviews['sentence_count'].std():.1f}")
        else:
            st.info("Dataset file not found. Please ensure 'Final_enhanced_dataset (1).csv' is in the same directory.")
    
    # ========== TAB 3: ABOUT ==========
    with tab3:
        st.header("About This Tool")
        
        st.markdown("""
        ## 🎯 Purpose
        This tool detects fake/AI-generated reviews by analyzing linguistic patterns 
        and textual characteristics that distinguish computer-generated content from 
        genuine human reviews.
        
        ## 📚 Training Data
        - **Dataset Size**: 44,194 reviews
        - **CG (Computer-Generated)**: AI-generated reviews
        - **OR (Original)**: Genuine human reviews
        - **Source**: Amazon product reviews with LLM-enhanced versions
        
        ## 🔬 Detection Method
        The model analyzes 9 key linguistic features:
        
        1. **Text Length** - CG reviews tend to be shorter
        2. **Vocabulary Richness** - CG has more repetition
        3. **Average Word Length** - Distinguishes writing patterns
        4. **Sentence Count** - CG often uses fewer sentences
        5. **Punctuation Ratio** - Different punctuation habits
        6. **Uppercase Ratio** - Capitalization patterns
        7. **Special Characters** - Symbol usage differences
        8. **Repetition Score** - Word repetition indicator
        9. **Average Sentence Length** - Sentence complexity
        
        ## 🏆 Model Performance
        - **Algorithm**: Random Forest Classifier (100 trees)
        - **Features**: 9 linguistic features
        - **Optimization**: Hyperparameters tuned for accuracy
        
        ## ⚠️ Limitations
        - Works best with reviews of reasonable length (>50 chars)
        - May have lower accuracy on heavily edited reviews
        - Best used as a supplementary detection tool
        
        ## 📖 How to Use
        1. Go to the **"Detect Review"** tab
        2. Paste or type a review
        3. Click **"Analyze"**
        4. Review the results and risk factors
        
        ## 💡 Interpretation
        - **High Fake Probability**: Likely AI-generated
        - **High Real Probability**: Likely genuine
        - **Risk Factors**: Specific patterns suggesting fakeness
        
        ---
        **Version**: 1.0 | **Last Updated**: 2024
        """)

if __name__ == "__main__":
    main()
