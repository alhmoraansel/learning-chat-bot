import sqlite3
import json
import numpy as np
import pickle
import os
import random
import time
import re
from datetime import datetime

# --- THIRD PARTY AI LIBS ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ImportError:
    print("CRITICAL MISSING LIBS: Run 'pip install sentence-transformers scikit-learn numpy'")
    exit()

# --- CONFIGURATION ---
DB_FILE = 'neural_bot_memory.db'
INTENT_MODEL_FILE = 'intent_model.pkl'
PCA_MODEL_FILE = 'pca_model.pkl'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
CONTEXT_TIMEOUT_SECONDS = 180
MAX_TRAINING_SAMPLES = 100 

# Tuning Thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.85,
    'MEDIUM': 0.75,
    'LOW': 0.65,
    'INTENT_MIN': 0.60,
    'CONTEXT_MIN': 0.70
}

# --- 1. HELPERS: LINGUISTICS & SAFETY ---
class SafetyGuard:
    PROFANITY_LIST = ["badword1", "badword2"] 
    
    @staticmethod
    def is_safe(text):
        if len(text) > 500: return False 
        if any(bad in text.lower() for bad in SafetyGuard.PROFANITY_LIST):
            return False
        return True

    @staticmethod
    def normalize_text(text):
        """Canonicalization: lowercase, strip punctuation, clean spaces"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text)

class Linguistics:
    """Detects sentence structures to filter learning garbage."""
    # Standard Question starters
    STARTERS = {"who", "what", "where", "when", "why", "how", "is", "are", "do", "does", "can", "could", "would", "should", "whats", "what's"}
    # Greetings/Conversational openers (Allow teaching these too)
    GREETINGS = {"hey", "hi", "hello", "sup", "yo", "greetings", "morning", "afternoon", "evening"}
    
    @staticmethod
    def is_teachable(text):
        text = text.strip().lower()
        # 1. Ends in question mark?
        if text.endswith("?"): return True
        
        parts = text.split(' ')
        if not parts: return False
        first_word = parts[0]
        
        # 2. Starts with question word?
        if first_word in Linguistics.STARTERS:
            return True
            
        # 3. Starts with greeting? (Fix for "hey", "sup")
        if first_word in Linguistics.GREETINGS:
            return True
            
        return False

class FactExtractor:
    """Robust rule-based extraction for user profile facts."""
    PATTERNS = [
        # "My [key] is [value]"
        (r"(?:my|the)\s+(favorite\s+\w+|\w+)\s+is\s+(.+)", "standard"),
        # "I am [value]" -> key=name/role
        (r"i am\s+(.+)", "identity"),
        # "I like [value]" -> key=likes
        (r"i like\s+(.+)", "likes"),
        # "Call me [value]" -> key=name
        (r"call me\s+(.+)", "name")
    ]

    @staticmethod
    def extract(text):
        text = text.strip().lower()
        for pat, type_ in FactExtractor.PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                if type_ == "standard":
                    return match.group(1).strip(), match.group(2).strip()
                elif type_ == "identity":
                    return "identity", match.group(1).strip()
                elif type_ == "likes":
                    return "likes", match.group(1).strip()
                elif type_ == "name":
                    return "name", match.group(1).strip()
        return None, None

# --- 2. THE NEURAL BRAIN ---
class NeuralBrain:
    def __init__(self):
        print("BRAIN: Loading Sentence-BERT...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.intent_classifier = None
        self.pca = None
        
        self.load_pca()
        self.load_or_train_intent_model()

    def get_embedding(self, text):
        """
        ALWAYS returns the RAW float32 embedding (384-dim).
        """
        vec = self.encoder.encode([text])[0].astype(np.float32)
        return vec

    def apply_pca(self, vectors):
        """Helper to transform raw vectors using loaded PCA."""
        if self.pca and len(vectors) > 0:
            # Handle single vector or matrix
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            return self.pca.transform(vectors).astype(np.float32)
        return vectors

    def load_pca(self):
        if os.path.exists(PCA_MODEL_FILE):
            try:
                with open(PCA_MODEL_FILE, 'rb') as f:
                    self.pca = pickle.load(f)
                print(f"BRAIN: Loaded PCA model (components={self.pca.n_components_})")
            except Exception as e:
                print(f"BRAIN: PCA load failed ({e}). Ignoring.")
                self.pca = None

    def train_pca(self, vectors):
        if len(vectors) < 50:
            print("BRAIN: Not enough data for PCA (Need 50+ vectors).")
            return

        print(f"BRAIN: Training PCA on {len(vectors)} vectors...")
        self.pca = PCA(n_components=min(128, len(vectors), vectors.shape[1]))
        self.pca.fit(vectors)
        
        with open(PCA_MODEL_FILE, 'wb') as f:
            pickle.dump(self.pca, f)
        print("BRAIN: PCA retrained and saved.")

    def load_or_train_intent_model(self, force_retrain=False, new_data=None):
        if not force_retrain and os.path.exists(INTENT_MODEL_FILE):
            try:
                with open(INTENT_MODEL_FILE, 'rb') as f:
                    self.intent_classifier = pickle.load(f)
                return
            except Exception:
                pass

        print("BRAIN: Training/Retraining Neural Intent Classifier...")
        
        # Base Dataset
        data = {
            "GREETING": ["hi", "hello", "hey", "good morning", "sup", "greetings", "yo"],
            "FAREWELL": ["bye", "goodbye", "quit", "exit", "see ya", "later"],
            "FACT_QUERY": ["what is my name", "who am i", "do you know me", "what do you know about X", "recall my data", "tell me about me"],
            "FACT_TEACH": ["my name is bob", "i like pizza", "my favorite color is blue", "remember that X is Y", "note that", "call me X"],
            "META": ["debug", "show memory", "clear cache", "help", "system status", "optimize"],
            "QA_SEARCH": ["what is the sun", "who is president", "how does code work", "meaning of life", "why is the sky blue"] 
        }

        X_text = []
        y_labels = []

        # 1. Load Base Data
        for label, phrases in data.items():
            X_text.extend(phrases)
            y_labels.extend([label] * len(phrases))

        # 2. Merge New Data (with Downsampling)
        if new_data:
            new_data_dict = {}
            for text, label in new_data:
                if label not in new_data_dict: new_data_dict[label] = []
                new_data_dict[label].append(text)
            
            for label, phrases in new_data_dict.items():
                if len(phrases) > MAX_TRAINING_SAMPLES:
                    phrases = random.sample(phrases, MAX_TRAINING_SAMPLES)
                X_text.extend(phrases)
                y_labels.extend([label] * len(phrases))

        X_vectors = self.encoder.encode(X_text)
        self.intent_classifier = LogisticRegression(random_state=42, max_iter=200)
        self.intent_classifier.fit(X_vectors, y_labels)
        
        with open(INTENT_MODEL_FILE, 'wb') as f:
            pickle.dump(self.intent_classifier, f)

    def predict_intent(self, vector):
        vector = vector.reshape(1, -1)
        pred = self.intent_classifier.predict(vector)[0]
        probs = self.intent_classifier.predict_proba(vector)[0]
        return pred, max(probs)

# --- 3. STORAGE ENGINE ---
class StorageEngine:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS qa_memory (
                id INTEGER PRIMARY KEY,
                question TEXT,
                question_norm TEXT UNIQUE, -- Canonical form
                answer TEXT,
                embedding BLOB,
                created_at REAL,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        cur.execute('CREATE TABLE IF NOT EXISTS intent_training_data (id INTEGER PRIMARY KEY, text TEXT, label TEXT, timestamp REAL)')
        cur.execute('CREATE TABLE IF NOT EXISTS user_facts (key TEXT PRIMARY KEY, value TEXT, confidence REAL DEFAULT 1.0, last_updated REAL)')
        cur.execute('CREATE TABLE IF NOT EXISTS conversation_log (id INTEGER PRIMARY KEY, turn_id INTEGER, parent_turn_id INTEGER, role TEXT, message TEXT, embedding BLOB, timestamp REAL)')

        self._migrate_schema(cur)
        self.conn.commit()

    def _migrate_schema(self, cur):
        cur.execute("PRAGMA table_info(qa_memory)")
        cols = [c[1] for c in cur.fetchall()]
        if 'question_norm' not in cols:
            cur.execute("ALTER TABLE qa_memory ADD COLUMN question_norm TEXT")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_q_norm ON qa_memory(question_norm)")

        cur.execute("PRAGMA table_info(conversation_log)")
        cols = [c[1] for c in cur.fetchall()]
        if 'turn_id' not in cols:
            cur.execute("ALTER TABLE conversation_log ADD COLUMN turn_id INTEGER")
            cur.execute("ALTER TABLE conversation_log ADD COLUMN parent_turn_id INTEGER")

    def save_qa(self, question, answer, vector):
        q_norm = SafetyGuard.normalize_text(question)
        cur = self.conn.cursor()
        vector = vector.astype(np.float32)
        
        try:
            cur.execute(
                "INSERT INTO qa_memory (question, question_norm, answer, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                (question, q_norm, answer, vector.tobytes(), time.time())
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"DB: Duplicate question detected ('{q_norm}'). Skipping.")
            return False

    def save_fact(self, key, value):
        cur = self.conn.cursor()
        cur.execute('INSERT OR REPLACE INTO user_facts (key, value, last_updated) VALUES (?, ?, ?)', 
                   (key, value, time.time()))
        self.conn.commit()

    def get_all_qa_vectors(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, embedding, answer FROM qa_memory")
        rows = cur.fetchall()
        
        ids, vectors, answers = [], [], []
        for row in rows:
            if row['embedding']:
                ids.append(row['id'])
                vectors.append(np.frombuffer(row['embedding'], dtype=np.float32))
                answers.append(row['answer'])
        
        if vectors:
            return ids, np.vstack(vectors), answers
        return [], np.empty((0, 384)), []

    def get_last_context_vector(self, time_window=180):
        cur = self.conn.cursor()
        cur.execute("SELECT embedding, timestamp FROM conversation_log WHERE role='user' ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        
        if row:
            ts = row['timestamp']
            if isinstance(ts, str): # Legacy check
                try: ts = datetime.fromisoformat(ts).timestamp()
                except: ts = 0.0
            
            if time.time() - ts < time_window:
                if row['embedding']: 
                    return np.frombuffer(row['embedding'], dtype=np.float32)
        return None

    def log_turn(self, turn_id, user_text, user_vec, bot_text):
        user_blob = user_vec.astype(np.float32).tobytes() if user_vec is not None else None
        now = time.time()
        
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.execute(
                "INSERT INTO conversation_log (turn_id, role, message, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
                (turn_id, 'user', user_text, user_blob, now)
            )
            if bot_text:
                self.conn.execute(
                    "INSERT INTO conversation_log (turn_id, role, message, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (turn_id, 'bot', bot_text, None, now)
                )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"DB: Log Error {e}")

    def save_intent_training_sample(self, text, label):
        self.conn.execute("INSERT INTO intent_training_data (text, label, timestamp) VALUES (?, ?, ?)", 
                          (text, label, time.time()))
        self.conn.commit()

    def prune_intent_data(self, max_per_label=50):
        cur = self.conn.cursor()
        labels = [row[0] for row in cur.execute("SELECT DISTINCT label FROM intent_training_data").fetchall()]
        
        for label in labels:
            cur.execute(f'''
                DELETE FROM intent_training_data 
                WHERE label = ? AND id NOT IN (
                    SELECT id FROM intent_training_data 
                    WHERE label = ? 
                    ORDER BY id DESC LIMIT ?
                )
            ''', (label, label, max_per_label))
        self.conn.commit()

    def get_intent_training_data(self):
        cur = self.conn.cursor()
        cur.execute("SELECT text, label FROM intent_training_data")
        return [(row['text'], row['label']) for row in cur.fetchall()]

# --- 4. THE NEURAL BOT (PIPELINE) ---
class NeuralBot:
    def __init__(self):
        self.db = StorageEngine(DB_FILE)
        self.brain = NeuralBrain()
        self.turn_counter = 0

    def check_retrain_trigger(self):
        data = self.db.get_intent_training_data()
        if len(data) > 20: 
            self.db.prune_intent_data(max_per_label=50) 
            self.brain.load_or_train_intent_model(force_retrain=True, new_data=data)

    def reflect_and_consolidate(self):
        print("SYSTEM: Running maintenance...")
        ids, vectors, answers = self.db.get_all_qa_vectors()
        
        if len(vectors) > 50 and (not self.brain.pca):
             self.brain.train_pca(vectors)

    # --- LAYER 1: PERCEPTION ---
    def _layer_1_perception(self, user_input):
        raw_vec = self.brain.get_embedding(user_input)
        
        state = {
            "text": user_input,
            "raw_vector": raw_vec,
            "intent": "UNKNOWN",
            "intent_conf": 0.0,
            "context_score": 0.0
        }

        # Intent Classification
        pred, conf = self.brain.predict_intent(raw_vec)
        
        if conf < CONFIDENCE_THRESHOLDS['INTENT_MIN']:
            state['intent'] = "UNKNOWN"
            state['intent_conf'] = conf
        else:
            state['intent'] = pred
            state['intent_conf'] = conf
            if conf > 0.95:
                self.db.save_intent_training_sample(user_input, pred)

        # Context Analysis
        last_vec_raw = self.db.get_last_context_vector()
        if last_vec_raw is not None:
            # FIX: Reshape for Sklearn (Ensure 2D)
            curr_compare = self.brain.apply_pca(state['raw_vector'])
            last_compare = self.brain.apply_pca(last_vec_raw)
            
            if len(curr_compare.shape) == 1: curr_compare = curr_compare.reshape(1, -1)
            if len(last_compare.shape) == 1: last_compare = last_compare.reshape(1, -1)
            
            sim = cosine_similarity(curr_compare, last_compare)[0][0]
            state['context_score'] = float(sim)
        
        return state

    # --- LAYER 2: DECISION ---
    def _layer_2_decision(self, state):
        intent = state['intent']
        text = state['text']

        if intent == "FAREWELL": return {"action": "EXIT", "response": "Goodbye!"}
        if intent == "GREETING": return {"action": "RESPOND", "response": "Hello!"}
        if intent == "COMPLAINT": return {"action": "RESPOND", "response": "Apologies."}

        # Fact Extraction
        if intent == "FACT_TEACH" or "my" in text or "i am" in text:
            key, val = FactExtractor.extract(text)
            if key and val:
                return {"action": "SAVE_FACT", "key": key, "val": val}

        if intent == "FACT_QUERY": 
             return {"action": "QUERY_PROFILE", "text": text}

        # QA Search
        ids, vectors, answers = self.db.get_all_qa_vectors()
        if len(ids) > 0:
            db_compare = self.brain.apply_pca(vectors)
            query_compare = self.brain.apply_pca(state['raw_vector'])
            
            # FIX: Reshape for Sklearn (Ensure 2D)
            if len(query_compare.shape) == 1:
                query_compare = query_compare.reshape(1, -1)
            
            sims = cosine_similarity(query_compare, db_compare)[0]
            best_idx = sims.argsort()[-1]
            best_sim = sims[best_idx]
            
            final_score = (0.7 * best_sim) + (0.3 * state['context_score'])
            
            if final_score > CONFIDENCE_THRESHOLDS['HIGH']:
                return {"action": "RESPOND", "response": answers[best_idx]}
            elif final_score > CONFIDENCE_THRESHOLDS['LOW']:
                return {"action": "RESPOND", "response": f"Did you mean: {answers[best_idx]}?"}
        
        return {"action": "FALLBACK_LEARN", "vector": state['raw_vector']}

    # --- LAYER 3: EXECUTION ---
    def _layer_3_execution(self, plan, state):
        response = None
        
        if plan['action'] == "EXIT":
            print(f"Bot: {plan['response']}")
            return False

        elif plan['action'] == "RESPOND":
            response = plan['response']
            print(f"Bot: {response}")

        elif plan['action'] == "SAVE_FACT":
            self.db.save_fact(plan['key'], plan['val'])
            response = f"Updated: {plan['key']} = {plan['val']}"
            print(f"Bot: {response}")

        elif plan['action'] == "QUERY_PROFILE":
            response = "I need better NER to answer that specific query."
            print(f"Bot: {response}")

        elif plan['action'] == "FALLBACK_LEARN":
            # UPDATED: Using is_teachable (includes greetings/whats)
            if not Linguistics.is_teachable(state['text']):
                response = "I don't know that."
                print(f"Bot: {response}")
            else:
                ans = input("Bot: Teach me? (Answer or [s]kip): ")
                if ans.lower() != 's' and len(ans) > 2:
                    self.db.save_qa(state['text'], ans, plan['vector'])
                    response = "Learned."
                else:
                    response = "Skipped."

        self.db.log_turn(self.turn_counter, state['text'], state['raw_vector'], response)
        return True

    def run(self):
        print("\n--- NEURAL BOT v4.2 (Robust) ---")
        while True:
            u_in = input("\nYou: ").strip()
            if not u_in: continue
            if u_in == 'quit': 
                self.reflect_and_consolidate()
                break
            
            self.turn_counter += 1
            state = self._layer_1_perception(u_in)
            plan = self._layer_2_decision(state)
            if not self._layer_3_execution(plan, state): break
            
            if self.turn_counter % 10 == 0: self.check_retrain_trigger()

if __name__ == '__main__':
    bot = NeuralBot()
    bot.run()