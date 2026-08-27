"""
⚙️ CONFIGURATION FILE FOR FAKE REVIEW DETECTOR
Customize detection parameters and thresholds
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Random Forest Hyperparameters
    'n_estimators': 100,        # Number of trees in forest
    'max_depth': 15,            # Maximum tree depth
    'min_samples_split': 10,    # Min samples to split node
    'min_samples_leaf': 5,      # Min samples in leaf node
    'random_state': 42,         # For reproducibility
    'n_jobs': -1,               # Use all available cores
}

# ============================================================================
# FEATURE THRESHOLDS FOR RISK DETECTION
# ============================================================================

RISK_THRESHOLDS = {
    # Text characteristics
    'min_text_length': 100,           # Minimum acceptable review length
    'max_short_review_length': 150,   # What we consider "too short"
    
    # Vocabulary analysis
    'min_vocab_richness': 0.70,       # Minimum vocabulary richness
    'low_vocab_threshold': 0.65,      # Very low vocabulary
    
    # Word characteristics
    'min_avg_word_length': 4.5,       # Minimum average word length
    'low_word_length_threshold': 4.0,
    
    # Sentence structure
    'min_sentence_count': 2,          # Minimum sentences
    'min_avg_sentence_length': 8,     # Minimum words per sentence
    
    # Punctuation patterns
    'max_punctuation_ratio': 0.15,    # Max punctuation density
    'high_punctuation_ratio': 0.20,
    
    # Capitalization patterns
    'max_uppercase_ratio': 0.10,      # Max uppercase letters
    'high_uppercase_ratio': 0.15,
    
    # Repetition analysis
    'max_repetition_score': 0.30,     # Max word repetition
    'high_repetition_score': 0.40,
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    'high_certainty': 0.75,      # >75% confidence threshold
    'medium_certainty': 0.50,    # >50% confidence threshold
    'low_certainty': 0.30,       # Lower bound for confidence
}

# ============================================================================
# UI SETTINGS
# ============================================================================

UI_CONFIG = {
    # Streamlit page configuration
    'page_title': '🔍 Fake Review Detector',
    'page_icon': '🔍',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    
    # Styling
    'theme': 'light',  # or 'dark'
    'accent_color': '#667eea',
    
    # Text area height
    'input_height': 200,
    
    # Max characters to display in preview
    'max_preview_chars': 100,
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    'csv_path': 'Final_enhanced_dataset (1).csv',
    'model_save_path': 'fake_review_model.pkl',
    'required_columns': [
        'category',
        'rating',
        'label',
        'text_',
        'label_binary',
        'vocab_richness',
        'avg_word_len',
        'sentence_count',
        'text_length'
    ],
}

# ============================================================================
# FEATURE EXTRACTION CONFIG
# ============================================================================

FEATURE_NAMES = {
    'text_length': 'Text Length (characters)',
    'vocab_richness': 'Vocabulary Richness',
    'avg_word_len': 'Average Word Length',
    'sentence_count': 'Sentence Count',
    'punctuation_ratio': 'Punctuation Ratio',
    'uppercase_ratio': 'Uppercase Ratio',
    'special_char_ratio': 'Special Character Ratio',
    'repetition_score': 'Repetition Score',
    'avg_sentence_length': 'Average Sentence Length',
}

FEATURE_DESCRIPTIONS = {
    'text_length': 'Total number of characters in the review',
    'vocab_richness': 'Ratio of unique words to total words (0-1)',
    'avg_word_len': 'Average number of characters per word',
    'sentence_count': 'Number of sentences in the review',
    'punctuation_ratio': 'Ratio of punctuation marks to total characters',
    'uppercase_ratio': 'Ratio of uppercase letters to total characters',
    'special_char_ratio': 'Ratio of special characters to total characters',
    'repetition_score': 'Indicator of how many words are repeated',
    'avg_sentence_length': 'Average number of words per sentence',
}

# ============================================================================
# RISK FACTOR MESSAGES
# ============================================================================

RISK_MESSAGES = {
    'short_review': '⚠️ Very short review (typical of CG)',
    'low_vocab': '⚠️ Low vocabulary richness (possible repetition)',
    'short_words': '⚠️ Short average word length',
    'few_sentences': '⚠️ Very few sentences',
    'high_punctuation': '⚠️ High punctuation density',
    'excessive_caps': '⚠️ Excessive uppercase letters',
    'high_repetition': '⚠️ High word repetition',
    'short_sentences': '⚠️ Very short sentences',
}

# ============================================================================
# DETECTION CATEGORIES
# ============================================================================

PREDICTION_LABELS = {
    0: {
        'label': '✅ GENUINE',
        'short': 'OR (Original)',
        'emoji': '✅',
        'color': 'green',
        'description': 'Likely a genuine human-written review'
    },
    1: {
        'label': '🤖 AI-GENERATED',
        'short': 'CG (Computer-Generated)',
        'emoji': '🤖',
        'color': 'red',
        'description': 'Likely an AI-generated fake review'
    }
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'detector.log',
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

PERFORMANCE_CONFIG = {
    'cache_predictions': True,      # Cache model predictions
    'cache_size_mb': 50,            # Max cache size
    'batch_size': 32,               # Batch processing size
    'max_reviews_batch': 1000,      # Max reviews in batch
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

ADVANCED_CONFIG = {
    # Feature importance
    'show_feature_importance': True,
    
    # ROC curve
    'show_roc_curve': False,
    
    # Confusion matrix
    'show_confusion_matrix': False,
    
    # Detailed statistics
    'show_detailed_stats': True,
    
    # Export options
    'allow_export': True,
    'export_formats': ['csv', 'json', 'txt'],
}

# ============================================================================
# TRAINING DATA STATISTICS (Reference)
# ============================================================================

DATASET_STATISTICS = {
    'total_reviews': 44194,
    'cg_reviews': 21597,      # Computer-generated
    'or_reviews': 22597,      # Original/Genuine
    
    'cg_stats': {
        'avg_length': 340,
        'std_length': 120,
        'avg_vocab': 0.75,
        'avg_sentences': 3.5,
    },
    
    'or_stats': {
        'avg_length': 450,
        'std_length': 150,
        'avg_vocab': 0.78,
        'avg_sentences': 4.2,
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_threshold(feature_name):
    """Get risk threshold for a specific feature"""
    return RISK_THRESHOLDS.get(feature_name, None)

def get_feature_description(feature_name):
    """Get description for a feature"""
    return FEATURE_DESCRIPTIONS.get(feature_name, 'No description available')

def get_prediction_info(prediction_class):
    """Get prediction information"""
    return PREDICTION_LABELS.get(prediction_class, {})

def is_high_confidence(confidence_value):
    """Check if confidence is high"""
    return confidence_value > CONFIDENCE_THRESHOLDS['high_certainty']

def is_medium_confidence(confidence_value):
    """Check if confidence is medium"""
    return CONFIDENCE_THRESHOLDS['medium_certainty'] <= confidence_value <= CONFIDENCE_THRESHOLDS['high_certainty']

def is_low_confidence(confidence_value):
    """Check if confidence is low"""
    return confidence_value < CONFIDENCE_THRESHOLDS['medium_certainty']

# ============================================================================
# CUSTOMIZATION EXAMPLES
# ============================================================================

"""
To customize the detector, modify these configs:

1. Adjust Risk Thresholds:
   RISK_THRESHOLDS['min_text_length'] = 80  # Lower threshold

2. Change Model Parameters:
   MODEL_CONFIG['n_estimators'] = 200  # More trees

3. Modify Feature Names:
   FEATURE_NAMES['vocab_richness'] = 'Word Diversity'

4. Adjust Confidence Thresholds:
   CONFIDENCE_THRESHOLDS['high_certainty'] = 0.80

5. Enable Advanced Features:
   ADVANCED_CONFIG['show_feature_importance'] = True
"""
