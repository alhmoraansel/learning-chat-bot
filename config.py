import os

# Paths
DB_FILE = 'neural_bot_memory.db'
INTENT_MODEL_FILE = 'intent_model.pkl'
ACT_MODEL_FILE = 'act_model.pkl'
PCA_MODEL_FILE = 'pca_model.pkl'
TOPIC_MODEL_FILE = 'topic_model.pkl'

# Models
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
GENERATIVE_MODEL_NAME = 'distilgpt2'

# Settings
CONTEXT_TIMEOUT_SECONDS = 300
ENABLE_LOGGING = True
ENABLE_GENERATIVE = False 
ENABLE_DEBUG_THOUGHTS = True

# Tuning Weights
W_SEMANTIC = 0.5
W_CONTEXT = 0.2
W_RECENCY = 0.1
W_PROFILE = 0.2

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.85,
    'MEDIUM': 0.75,
    'LOW': 0.60,
    'INTENT_MIN': 0.55,
}