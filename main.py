import sqlite3
import json
import numpy as np
import pickle
import os
import random
import time
import re
import shutil
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. DEPENDENCY CHECK & MOCKS
# ==========================================
TRANSFORMERS_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    try:
        from transformers import pipeline
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        pass
    print("SUCCESS: AI Libraries loaded.")
except ImportError:
    print("\nWARNING: AI libraries (sentence-transformers, sklearn) not found.")
    print("Running in LIGHT MODE (Rule-based only).")
    print("To enable full AI, run: pip install sentence-transformers scikit-learn numpy transformers\n")

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_FILE = 'neural_bot_memory.db'
INTENT_MODEL_FILE = 'intent_model.pkl'
ACT_MODEL_FILE = 'act_model.pkl'
PCA_MODEL_FILE = 'pca_model.pkl'
TOPIC_MODEL_FILE = 'topic_model.pkl'

# Models
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATIVE_MODEL_NAME = 'google/flan-t5-small'

# Settings
CONTEXT_TIMEOUT_SECONDS = 300
ENABLE_LOGGING = True
ENABLE_GENERATIVE = True
ENABLE_DEBUG_THOUGHTS = True

# Tuning Weights (Hybrid Search)
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

# ==========================================
# 2. DATA STRUCTURES
# ==========================================
@dataclass
class TopicNode:
    id: int
    centroid: np.ndarray
    last_active: float
    label: str = "general"

@dataclass
class UserProfile:
    facts: Dict[str, str] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

@dataclass
class WorkingMemory:
    active_topic: Optional[TopicNode] = None
    last_turn_vector: Optional[np.ndarray] = None
    last_memory_vector: Optional[np.ndarray] = None
    pending_correction: Optional[Dict] = None
    pending_fact: Optional[Dict] = None
    history_window: List[Dict] = field(default_factory=list)

@dataclass
class PipelineState:
    raw_text: str
    norm_text: str = ""
    timestamp: float = 0.0
    
    # Perception
    raw_vector: Optional[np.ndarray] = None
    features: Dict[str, bool] = field(default_factory=dict)
    
    # Understanding
    intent: str = "UNKNOWN"
    intent_conf: float = 0.0
    act: str = "UNKNOWN"
    act_conf: float = 0.0
    
    # Context & Profile
    query_vector: Optional[np.ndarray] = None
    weighted_context: Optional[np.ndarray] = None
    user_profile: Optional[UserProfile] = None
    
    # Reasoning (The Planner)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    thought_trace: List[str] = field(default_factory=list)
    
    # Execution
    response: Optional[str] = None
    stop_pipeline: bool = False

# ==========================================
# 3. UTILITIES
# ==========================================
class BackupManager:
    @staticmethod
    def create_backup():
        if os.path.exists(DB_FILE):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{DB_FILE}.{timestamp}.bak"
            try:
                backups = sorted([f for f in os.listdir('.') if f.startswith(DB_FILE) and f.endswith('.bak')])
                if len(backups) >= 3:
                    try: os.remove(backups[0])
                    except: pass
                shutil.copy2(DB_FILE, backup_name)
            except: pass

class SafetyGuard:
    PROFANITY_LIST = ["badword1", "badword2"] # Expand as needed
    @staticmethod
    def is_safe(text):
        if len(text) > 500: return False
        if any(bad in text.lower() for bad in SafetyGuard.PROFANITY_LIST): return False
        return True

class Linguistics:
    STARTERS = {"who", "what", "where", "when", "why", "how", "is", "are", "do", "does", "can", "could", "would", "should", "whats", "what's"}
    GREETINGS = {"hey", "hi", "hello", "sup", "yo", "greetings", "morning"}
    
    SYNONYMS = {
        "basically": ["essentially", "in short", "fundamentally"],
        "think": ["believe", "suspect", "guess"],
        "usually": ["typically", "generally", "often"],
        "correct": ["accurate", "right", "spot on"],
        "wrong": ["incorrect", "off", "inaccurate"]
    }

    @staticmethod
    def paraphrase(text):
        words = text.split()
        new_words = []
        for w in words:
            clean_w = re.sub(r'[^\w]', '', w.lower())
            if clean_w in Linguistics.SYNONYMS and random.random() > 0.6:
                replacement = random.choice(Linguistics.SYNONYMS[clean_w])
                if w[0].isupper(): replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(w)
        return " ".join(new_words)

class Personality:
    TEMPLATES = {
        "GIVE_INFO": ["Here's what I know: {}.", "Basically: {}.", "The answer is: {}.", "As far as I recall: {}."],
        "UNCERTAIN": ["I'm not 100% sure, but: {}.", "It might be related to this: {}.", "Take this with a grain of salt: {}."],
        "CLARIFY": ["Do you mean {A} or {B}?", "Wait, are we talking about {topic}?", "Could you rephrase that?"],
        "REPAIR": ["My bad. Let me relearn that.", "I misunderstood. What is the correct answer?", "Thanks for the correction. What should it be?"],
        "SMALLTALK": ["Glad to hear it!", "Awesome.", "Great!", "Happy to help.", "Anytime!", "Good to know."]
    }
    
    @staticmethod
    def format(act, content):
        if act not in Personality.TEMPLATES: return content
        tmpl = random.choice(Personality.TEMPLATES[act])
        if len(content.split()) > 3 and act != "SMALLTALK":
            content = Linguistics.paraphrase(content)
        if "{}" in tmpl: return tmpl.format(content)
        return tmpl

# ==========================================
# 4. BRAIN COMPONENTS (Encoders & Classifiers)
# ==========================================
class Encoder:
    def __init__(self):
        self.model = None
        if SKLEARN_AVAILABLE:
            print(f"BRAIN: Loading Encoder ({EMBEDDING_MODEL_NAME})...")
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as e:
                print(f"BRAIN: Failed to load SentenceTransformer: {e}")

    def encode(self, text: str) -> np.ndarray:
        if self.model:
            return self.model.encode([text])[0].astype(np.float32)
        else:
            # Dummy vector for Light Mode
            random.seed(len(text))
            return np.random.rand(384).astype(np.float32)

class Generator:
    def __init__(self):
        self.pipeline = None
        if ENABLE_GENERATIVE and TRANSFORMERS_AVAILABLE:
            try:
                print(f"BRAIN: Loading GenAI ({GENERATIVE_MODEL_NAME})...")
                self.pipeline = pipeline('text2text-generation', model=GENERATIVE_MODEL_NAME)
            except Exception as e:
                print(f"BRAIN: GenAI load failed: {e}")

    def generate(self, question, context=None):
        if not self.pipeline: return None
        try:
            if context:
                prompt = f"Answer based on context: {context}. Question: {question}"
            else:
                prompt = f"Answer this question: {question}"
            
            # --- FIX: ADDED REPETITION PENALTIES & FIXED LENGTH PARAM ---
            output = self.pipeline(
                prompt, 
                max_new_tokens=60,         # Fixes warning (replaces max_length)
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9,
                repetition_penalty=1.5,    # Prevents "symphony is a symphony" loop
                no_repeat_ngram_size=2     # Prevents 2-word phrase repetition
            )
            return output[0]['generated_text'].strip()
        except: return None

class IntentClassifier:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model = self._load()
        if not self.model and SKLEARN_AVAILABLE: 
            self.train_baseline()

    def _load(self):
        if os.path.exists(INTENT_MODEL_FILE):
            try:
                with open(INTENT_MODEL_FILE, 'rb') as f: return pickle.load(f)
            except: pass
        return None

    def train_baseline(self):
        print("BRAIN: Training robust Intent Classifier...")
        # Synthetic Data Generation
        entities = ["sun", "moon", "gravity", "python", "code", "life", "ai", "space", "time", "math", "cats", "food", "earth"]
        names = ["bob", "alice", "john", "jane", "user", "human"]
        foods = ["pizza", "pasta", "sushi", "burgers", "salad"]
        
        def generate_greeting():
            bases = ["hi", "hello", "hey", "sup", "yo", "greetings", "good morning"]
            return [f"{b}{p}" for b in bases for p in ["", "!", "?"]]

        def generate_farewell():
            bases = ["bye", "goodbye", "quit", "exit", "leave", "see ya"]
            return [f"{b}{p}" for b in bases for p in ["", "!", "."]]

        def generate_fact_teach():
            examples = []
            for n in names: examples.append(f"my name is {n}")
            for f in foods: examples.append(f"i like {f}")
            return examples * 15

        def generate_qa_search():
            starts = ["what is", "tell me about", "how does", "explain", "define", "who is", "why is"]
            examples = []
            for s in starts:
                for e in entities: examples.append(f"{s} {e}")
            return examples

        data = {
            "GREETING": generate_greeting(),
            "FAREWELL": generate_farewell(),
            "FACT_TEACH": generate_fact_teach(),
            "QA_SEARCH": generate_qa_search(),
            "META": ["debug", "show memory", "wipe data", "reset", "system status"] * 20,
            "FACT_QUERY": ["who am i", "what is my name", "do you know me"] * 20
        }

        X, y = [], []
        for label, phrases in data.items():
            phrases = list(set(phrases))
            X.extend(phrases)
            y.extend([label]*len(phrases))
        
        if not self.encoder.model: return

        vecs = self.encoder.model.encode(X)
        base_clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
        calibrated_clf.fit(vecs, y)
        self.model = calibrated_clf
        with open(INTENT_MODEL_FILE, 'wb') as f: pickle.dump(self.model, f)

    def predict(self, vector):
        if not self.model: return "UNKNOWN", 0.0
        vector = vector.reshape(1, -1)
        pred = self.model.predict(vector)[0]
        probs = self.model.predict_proba(vector)[0]
        return pred, max(probs)

class ActClassifier:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model = self._load()
        if not self.model and SKLEARN_AVAILABLE: 
            self.train_baseline()

    def _load(self):
        if os.path.exists(ACT_MODEL_FILE):
            try:
                with open(ACT_MODEL_FILE, 'rb') as f: return pickle.load(f)
            except: pass
        return None

    def train_baseline(self):
        print("BRAIN: Training robust Act Classifier...")
        def gen_affirm(): return ["yes", "yeah", "yep", "correct", "right", "true", "sure", "ok"] * 5
        def gen_deny(): return ["no", "nope", "wrong", "false", "incorrect", "negative", "not really"] * 5
        def gen_emotion(): return ["awesome", "cool", "great", "wow", "amazing", "nice", "perfect", "thanks"] * 5
        
        data = {
            "AFFIRM": gen_affirm(),
            "DENY": gen_deny(),
            "REQUEST_INFO": ["tell me", "what is", "explain", "describe"] * 5,
            "GIVE_INFO": ["it is", "because", "basically", "usually"] * 5,
            "CLARIFY": ["what?", "huh?", "pardon?", "excuse me?"] * 5,
            "FOLLOWUP": ["and", "but", "so", "then", "also"] * 5,
            "EMOTION": gen_emotion()
        }

        X, y = [], []
        for label, phrases in data.items():
            X.extend(phrases)
            y.extend([label]*len(phrases))
        
        if not self.encoder.model: return

        vecs = self.encoder.model.encode(X)
        base_clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
        calibrated_clf.fit(vecs, y)
        self.model = calibrated_clf
        with open(ACT_MODEL_FILE, 'wb') as f: pickle.dump(self.model, f)

    def predict(self, vector):
        if not self.model: return "UNKNOWN", 0.0
        vector = vector.reshape(1, -1)
        pred = self.model.predict(vector)[0]
        probs = self.model.predict_proba(vector)[0]
        return pred, max(probs)

class NeuralBrainService:
    def __init__(self):
        self.encoder = Encoder()
        self.intent_classifier = IntentClassifier(self.encoder)
        self.act_classifier = ActClassifier(self.encoder)
        self.generator = Generator()
        
        self.pca = self._load_pickle(PCA_MODEL_FILE)
        self.topic_model = self._load_pickle(TOPIC_MODEL_FILE)
        
        self.vector_cache = {"vectors": None, "answers": [], "ids": [], "timestamps": [], "usage": []}
        self.is_cache_dirty = True

    def _load_pickle(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f: return pickle.load(f)
            except: pass
        return None

    def get_embedding(self, text):
        return self.encoder.encode(text)

    def apply_pca(self, vectors):
        if self.pca and len(vectors) > 0 and SKLEARN_AVAILABLE:
            if len(vectors.shape) == 1: vectors = vectors.reshape(1, -1)
            return self.pca.transform(vectors).astype(np.float32)
        return vectors

    def train_pca(self, vectors):
        if not SKLEARN_AVAILABLE: return
        if len(vectors) < 50: return
        self.pca = PCA(n_components=min(128, len(vectors), vectors.shape[1]))
        self.pca.fit(vectors)
        with open(PCA_MODEL_FILE, 'wb') as f: pickle.dump(self.pca, f)
        self.is_cache_dirty = True

# ==========================================
# 5. MEMORY (Storage)
# ==========================================
class StorageService:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS qa_memory (
            id INTEGER PRIMARY KEY, question TEXT, question_norm TEXT UNIQUE, 
            answer TEXT, embedding BLOB, created_at REAL, usage_count INTEGER DEFAULT 0, flags TEXT)''')
        cur.execute('CREATE TABLE IF NOT EXISTS user_facts (key TEXT PRIMARY KEY, value TEXT, confidence REAL, last_updated REAL)')
        cur.execute('CREATE TABLE IF NOT EXISTS conversation_log (id INTEGER PRIMARY KEY, role TEXT, message TEXT, embedding BLOB, timestamp REAL)')
        
        cur.execute("PRAGMA table_info(qa_memory)")
        cols = [c[1] for c in cur.fetchall()]
        if 'flags' not in cols: cur.execute("ALTER TABLE qa_memory ADD COLUMN flags TEXT")
        self.conn.commit()

    def load_cache(self, brain):
        if not brain.is_cache_dirty and brain.vector_cache['vectors'] is not None: return
        cur = self.conn.cursor()
        cur.execute("SELECT id, embedding, answer, created_at, usage_count FROM qa_memory WHERE flags IS NULL OR flags != 'flagged'")
        rows = cur.fetchall()
        vecs, ans, ids, ts, usage = [], [], [], [], []
        for row in rows:
            if row['embedding']:
                vecs.append(np.frombuffer(row['embedding'], dtype=np.float32))
                ans.append(row['answer'])
                ids.append(row['id'])
                ts.append(row['created_at'])
                usage.append(row['usage_count'])
        
        if vecs:
            raw_matrix = np.vstack(vecs)
            brain.vector_cache = {
                "vectors": brain.apply_pca(raw_matrix),
                "answers": ans, "ids": ids,
                "timestamps": np.array(ts), "usage": np.array(usage)
            }
        else:
            brain.vector_cache = {"vectors": None, "answers": [], "ids": [], "timestamps": [], "usage": []}
        brain.is_cache_dirty = False

    def save_qa(self, q, a, vec):
        try:
            self.conn.execute("INSERT INTO qa_memory (question, question_norm, answer, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                (q, q.lower().strip(), a, vec.tobytes(), time.time()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError: return False

    def save_fact(self, k, v, conf=1.0):
        self.conn.execute("INSERT OR REPLACE INTO user_facts (key, value, confidence, last_updated) VALUES (?, ?, ?, ?)", 
                          (k, v, conf, time.time()))
        self.conn.commit()

    def get_user_facts(self):
        cur = self.conn.cursor()
        cur.execute("SELECT key, value FROM user_facts")
        return {row['key']: row['value'] for row in cur.fetchall()}

    def flag_memory(self, mem_id):
        self.conn.execute("UPDATE qa_memory SET flags = 'flagged' WHERE id = ?", (mem_id,))
        self.conn.commit()

    def log(self, role, msg, vec=None):
        if not ENABLE_LOGGING: return
        blob = vec.tobytes() if vec is not None else None
        self.conn.execute("INSERT INTO conversation_log (role, message, embedding, timestamp) VALUES (?, ?, ?, ?)", 
                          (role, msg, blob, time.time()))
        self.conn.commit()

    def wipe(self):
        self.conn.execute("DELETE FROM user_facts")
        self.conn.execute("DELETE FROM conversation_log")
        self.conn.execute("DELETE FROM qa_memory")
        self.conn.commit()

# ==========================================
# 6. PIPELINE LAYERS
# ==========================================
class InputLayer:
    @staticmethod
    def process(state: PipelineState) -> PipelineState:
        text = state.raw_text
        if len(text) > 500: 
            state.response = "Input too long."
            state.stop_pipeline = True
            return state
        norm = re.sub(r'[^\w\s]', '', text.lower().strip())
        state.norm_text = norm
        state.timestamp = time.time()
        return state

class PerceptionLayer:
    def __init__(self, brain: NeuralBrainService):
        self.brain = brain
        
    def process(self, state: PipelineState) -> PipelineState:
        state.raw_vector = self.brain.get_embedding(state.raw_text)
        txt = state.norm_text
        state.features = {
            "is_question": state.raw_text.strip().endswith("?") or txt.startswith(("what", "who", "how", "why", "is", "do", "can")),
            "is_followup": txt.startswith(("and", "but", "so")) or any(x in txt for x in ["it", "that", "he", "she"]),
            "is_contradiction": txt.startswith("no ") or txt == "no" or "that is wrong" in txt or "incorrect" in txt,
            "has_fact_pattern": "my name is" in txt or "i like" in txt or "i hate" in txt or "i love" in txt,
            "is_compound": " and " in txt
        }
        return state

class UnderstandingLayer:
    def __init__(self, brain: NeuralBrainService):
        self.brain = brain
        self.greetings = {"hey", "hi", "hello", "sup", "yo", "greetings", "morning"}

    def process(self, state: PipelineState) -> PipelineState:
        first_word = state.norm_text.split(' ')[0] if state.norm_text else ""
        if first_word in self.greetings and len(state.norm_text.split()) < 4:
            state.intent = "GREETING"
            state.intent_conf = 1.0
            state.thought_trace.append("Understanding: Greeting detected via rule.")
            return state

        i_pred, i_conf = self.brain.intent_classifier.predict(state.raw_vector)
        state.intent = i_pred if i_conf > CONFIDENCE_THRESHOLDS['INTENT_MIN'] else "UNKNOWN"
        state.intent_conf = i_conf
        
        a_pred, a_conf = self.brain.act_classifier.predict(state.raw_vector)
        state.act = a_pred
        state.act_conf = a_conf
        
        state.thought_trace.append(f"Understanding: Intent={state.intent}, Act={state.act}")
        return state

class ContextLayer:
    def __init__(self, memory: WorkingMemory, db: StorageService, brain: NeuralBrainService):
        self.mem = memory
        self.db = db
        self.brain = brain

    def process(self, state: PipelineState) -> PipelineState:
        facts = self.db.get_user_facts()
        profile_texts = []
        for k, v in facts.items():
            if k.startswith("preference:"):
                obj = k.split(":", 1)[1]
                profile_texts.append(f"{v} {obj}")
            else:
                profile_texts.append(f"{k} {v}")
        
        profile_vec = None
        if profile_texts:
            text_blob = " ".join(profile_texts)
            profile_vec = self.brain.get_embedding(text_blob)
        state.user_profile = UserProfile(facts=facts, interests=profile_texts, embedding=profile_vec)
        
        now = time.time()
        if self.mem.active_topic:
            if now - self.mem.active_topic.last_active > CONTEXT_TIMEOUT_SECONDS:
                self.mem.active_topic = None
            else:
                alpha = 0.8
                self.mem.active_topic.centroid = (alpha * self.mem.active_topic.centroid) + ((1-alpha) * state.raw_vector)
                self.mem.active_topic.last_active = now

        query = state.raw_vector
        if state.features["is_followup"] and self.mem.last_turn_vector is not None:
            query = (0.7 * state.raw_vector) + (0.3 * self.mem.last_turn_vector)
            state.thought_trace.append("Context: Merged follow-up vector.")
        
        state.query_vector = query
        state.weighted_context = self.mem.active_topic.centroid if self.mem.active_topic else None
        return state

class ReasoningLayer:
    def process(self, state: PipelineState) -> PipelineState:
        # 1. Fact Extraction
        if state.features["has_fact_pattern"] or state.intent == "FACT_TEACH":
            m_pref = re.search(r"\bi (like|love|hate|dislike)s? (.+)", state.raw_text, re.IGNORECASE)
            if m_pref:
                verb = m_pref.group(1).lower()
                obj = m_pref.group(2).strip()
                key = f"preference:{obj}"
                
                existing_val = state.user_profile.facts.get(key)
                if existing_val and existing_val != verb:
                    state.plan.append({
                        "op": "CONFIRM_FACT", 
                        "key": key, 
                        "val": verb, 
                        "old_val": existing_val, 
                        "obj_name": obj
                    })
                else:
                    state.plan.append({"op": "SAVE_FACT", "key": key, "val": verb})
                return state

            m_attr = re.search(r"my (\w+) is (.+)", state.raw_text, re.IGNORECASE)
            if m_attr:
                key = m_attr.group(1).lower()
                val = m_attr.group(2).strip()
                existing_val = state.user_profile.facts.get(key)
                if existing_val and existing_val != val:
                      state.plan.append({"op": "CONFIRM_FACT", "key": key, "val": val, "old_val": existing_val, "obj_name": key})
                else:
                    state.plan.append({"op": "SAVE_FACT", "key": key, "val": val})
                return state

        # 2. Self-Correction
        if state.features["is_contradiction"] and state.intent != "FACT_TEACH":
            state.thought_trace.append("Reasoning: Contradiction detected. Initiating repair.")
            state.plan.append({"op": "REPAIR_MEMORY"})
            return state

        # 3. Smalltalk / Emotion
        if (state.act == "AFFIRM" or state.act == "EMOTION") and state.intent == "UNKNOWN":
             state.plan.append({"op": "RESPOND", "text": "Glad to hear it!", "style": "SMALLTALK"})
             return state

        # 4. Compound Query
        if state.features["is_compound"] and state.intent == "QA_SEARCH":
            parts = state.raw_text.split(" and ")
            if len(parts) == 2:
                state.plan.append({"op": "RETRIEVE", "query_text": parts[0]})
                state.plan.append({"op": "RETRIEVE", "query_text": parts[1]})
                state.plan.append({"op": "SYNTHESIZE"}) 
                return state

        # 5. Default
        if state.intent == "QA_SEARCH" or state.act == "REQUEST_INFO":
            state.plan.append({"op": "RETRIEVE", "query_vector": state.query_vector})
        elif state.intent == "FAREWELL":
            state.plan.append({"op": "EXIT"})
        elif state.intent == "GREETING":
            state.plan.append({"op": "RESPOND", "text": "Hello! I am ready."})
        elif state.intent == "META" and "wipe" in state.norm_text:
            state.plan.append({"op": "WIPE"})
        else:
            state.plan.append({"op": "FALLBACK_TEACH"})
            
        return state

class RetrievalLayer:
    def __init__(self, brain: NeuralBrainService, db: StorageService):
        self.brain = brain
        self.db = db

    def process(self, state: PipelineState) -> PipelineState:
        new_plan = []
        retrieved_results = []
        
        for step in state.plan:
            if step["op"] == "RETRIEVE":
                self.db.load_cache(self.brain)
                cache = self.brain.vector_cache
                
                # Check empty DB
                if cache['vectors'] is None or len(cache['vectors']) == 0:
                    new_plan.append({"op": "LLM_FALLBACK", "query": step.get("query_text", state.raw_text)})
                    continue
                
                q_vec = step.get("query_vector")
                if q_vec is None and "query_text" in step:
                    q_vec = self.brain.get_embedding(step["query_text"])
                
                q_pca = self.brain.apply_pca(q_vec).reshape(1, -1)
                
                # Hybrid Search
                if SKLEARN_AVAILABLE:
                    sem_sims = cosine_similarity(q_pca, cache['vectors'])[0]
                    
                    ctx_sims = np.zeros_like(sem_sims)
                    if state.weighted_context is not None:
                        c_pca = self.brain.apply_pca(state.weighted_context).reshape(1, -1)
                        ctx_sims = cosine_similarity(c_pca, cache['vectors'])[0]
                    
                    prof_sims = np.zeros_like(sem_sims)
                    if state.user_profile and state.user_profile.embedding is not None:
                        p_pca = self.brain.apply_pca(state.user_profile.embedding).reshape(1, -1)
                        prof_sims = cosine_similarity(p_pca, cache['vectors'])[0]

                    now = time.time()
                    recency = 1.0 / (1.0 + np.log1p(np.maximum(0, now - cache['timestamps'])))
                    
                    scores = (W_SEMANTIC * sem_sims) + \
                             (W_CONTEXT * ctx_sims) + \
                             (W_RECENCY * recency) + \
                             (W_PROFILE * prof_sims)
                    
                    best_idx = scores.argsort()[-1]
                    best_score = scores[best_idx]
                    
                    state.thought_trace.append(f"Retrieval: Score={best_score:.2f}")
                    
                    if best_score > CONFIDENCE_THRESHOLDS['HIGH']:
                        res_text = cache['answers'][best_idx]
                        new_plan.append({"op": "RESPOND", "text": res_text, "style": "GIVE_INFO"})
                        retrieved_results.append(res_text)
                    elif best_score > CONFIDENCE_THRESHOLDS['LOW']:
                        res_text = cache['answers'][best_idx]
                        new_plan.append({"op": "RESPOND", "text": res_text, "uncertain": True, "style": "UNCERTAIN"})
                        retrieved_results.append(res_text)
                    else:
                        new_plan.append({"op": "LLM_FALLBACK", "query": step.get("query_text", state.raw_text)})
                else:
                    new_plan.append({"op": "LLM_FALLBACK", "query": step.get("query_text", state.raw_text)})
            
            elif step["op"] == "SYNTHESIZE":
                if retrieved_results:
                    combined = " ".join(retrieved_results)
                    new_plan.append({"op": "LLM_SYNTHESIZE", "context": combined, "query": state.raw_text})
                else:
                      new_plan.append({"op": "LLM_FALLBACK", "query": state.raw_text})
            else:
                new_plan.append(step)
        
        state.plan = new_plan
        return state

class ExecutionLayer:
    def __init__(self, db: StorageService, memory: WorkingMemory, sys_mode: dict, brain: NeuralBrainService):
        self.db = db
        self.mem = memory
        self.mode = sys_mode
        self.brain = brain

    def process(self, state: PipelineState) -> PipelineState:
        if self.mode["state"] == "CONFIRMING_FACT":
            if state.norm_text in ["yes", "yeah", "sure", "ok"]:
                self.db.save_fact(self.mem.pending_fact['key'], self.mem.pending_fact['val'])
                state.response = "Okay, updated your profile."
            else:
                state.response = "Got it, keeping the old setting."
            self.mode["state"] = "IDLE"
            self.mem.pending_fact = None
            return state

        if self.mode["state"] == "TEACHING":
            if state.norm_text in ["cancel", "stop", "no"]:
                state.response = "Cancelled."
                self.mode["state"] = "IDLE"
            else:
                self.mem.last_memory_vector = self.mem.pending_correction['v']
                self.db.save_qa(self.mem.pending_correction['q'], state.raw_text, self.mem.pending_correction['v'])
                self.brain.is_cache_dirty = True
                state.response = "Thanks, learned."
                self.mode["state"] = "IDLE"
            return state

        final_responses = []
        
        for step in state.plan:
            op = step["op"]
            
            if op == "EXIT":
                state.response = "Goodbye!"
                state.stop_pipeline = True
                
            elif op == "WIPE":
                self.db.wipe()
                final_responses.append("Memory wiped.")
                
            elif op == "SAVE_FACT":
                self.db.save_fact(step['key'], step['val'])
                k, v = step['key'], step['val']
                if k.startswith("preference:"):
                    obj = k.split(":", 1)[1]
                    final_responses.append(f"Noted that you {v} {obj}.")
                else:
                    final_responses.append(f"Noted: {k} is {v}.")
            
            elif op == "CONFIRM_FACT":
                self.mode["state"] = "CONFIRMING_FACT"
                self.mem.pending_fact = {"key": step['key'], "val": step['val']}
                if "preference" in step['key']:
                    state.response = f"You told me you {step['old_val']} {step['obj_name']}, but now you {step['val']} it. Are you sure?"
                else:
                    state.response = f"You previously said your {step['obj_name']} is {step['old_val']}. Update to {step['val']}?"
                return state

            elif op == "RESPOND":
                txt = step['text']
                style = step.get("style", "GIVE_INFO")
                if step.get("uncertain"): txt = f"I think: {txt}"
                
                formatted = Personality.format(style, txt)
                final_responses.append(formatted)
                
                if state.query_vector is not None:
                    self.mem.last_memory_vector = state.query_vector

            elif op == "LLM_FALLBACK":
                if self.brain.generator:
                    gen_ans = self.brain.generator.generate(step['query'])
                    if gen_ans:
                        final_responses.append(f"(Generated): {gen_ans}")
                    else:
                        if state.features["is_question"]:
                            self.mode["state"] = "TEACHING"
                            self.mem.pending_correction = {"q": state.raw_text, "v": state.raw_vector}
                            state.response = "I don't know that. Teach me? (or 'cancel')"
                            return state
                        else:
                            final_responses.append("I'm not sure.")
                else:
                    if state.features["is_question"]:
                        self.mode["state"] = "TEACHING"
                        self.mem.pending_correction = {"q": state.raw_text, "v": state.raw_vector}
                        state.response = "I don't know that. Teach me? (or 'cancel')"
                        return state
                    else:
                        final_responses.append("I'm not sure.")

            elif op == "LLM_SYNTHESIZE":
                if self.brain.generator:
                    gen_ans = self.brain.generator.generate(step['query'], context=step['context'])
                    if gen_ans:
                        final_responses.append(gen_ans)
                    else:
                        final_responses.append(f"Combined: {step['context']}")
                else:
                    final_responses.append(f"Combined: {step['context']}")

            elif op == "FALLBACK_TEACH":
                if state.features["is_question"]:
                    self.mode["state"] = "TEACHING"
                    self.mem.pending_correction = {"q": state.raw_text, "v": state.raw_vector}
                    state.response = "I don't know that yet. Teach me? (or 'cancel')"
                    return state 
                else:
                    final_responses.append("I'm not sure how to respond to that.")

            elif op == "REPAIR_MEMORY":
                # Find memory ID based on vector similarity
                self.db.load_cache(self.brain)
                cache = self.brain.vector_cache
                target_vec = self.mem.last_memory_vector
                
                found = False
                if target_vec is not None and cache['vectors'] is not None and SKLEARN_AVAILABLE:
                    q_pca = self.brain.apply_pca(target_vec).reshape(1, -1)
                    sims = cosine_similarity(q_pca, cache['vectors'])[0]
                    best_idx = sims.argsort()[-1]
                    if sims[best_idx] > 0.8:
                        mem_id = cache['ids'][best_idx]
                        self.db.flag_memory(mem_id)
                        self.brain.is_cache_dirty = True
                        found = True
                
                if found:
                      final_responses.append("I've flagged that memory as incorrect. What is the right answer?")
                      self.mode["state"] = "TEACHING"
                      self.mem.pending_correction = {"q": "REPAIR", "v": target_vec}
                      state.response = final_responses[-1]
                      return state
                else:
                      final_responses.append("I couldn't find the specific memory to fix.")

        if not state.response and final_responses:
            state.response = " ".join(final_responses)
            
        self.db.log("user", state.raw_text, state.raw_vector)
        if state.response: self.db.log("bot", state.response)
        
        self.mem.last_turn_vector = state.raw_vector
        if not self.mem.active_topic and state.raw_vector is not None:
             self.mem.active_topic = TopicNode(id=0, centroid=state.raw_vector, last_active=time.time())

        return state

# ==========================================
# 7. ORCHESTRATOR
# ==========================================
class PipelineBot:
    def __init__(self):
        BackupManager.create_backup()
        
        self.brain = NeuralBrainService()
        self.db = StorageService()
        self.memory = WorkingMemory()
        self.sys_mode = {"state": "IDLE"}
        
        self.layers = [
            InputLayer(),
            PerceptionLayer(self.brain),
            UnderstandingLayer(self.brain),
            ContextLayer(self.memory, self.db, self.brain),
            ReasoningLayer(),
            RetrievalLayer(self.brain, self.db),
            ExecutionLayer(self.db, self.memory, self.sys_mode, self.brain)
        ]

    def run_turn(self, user_input: str) -> bool:
        state = PipelineState(raw_text=user_input)
        
        for layer in self.layers:
            state = layer.process(state)
            if state.stop_pipeline: break
        
        if ENABLE_DEBUG_THOUGHTS and state.thought_trace:
            print(f"\n[THOUGHTS]: {state.thought_trace}")
            
        if state.response:
            print(f"Bot: {state.response}")
            
        return not state.stop_pipeline

    def start(self):
        print("\n--- NEURAL AGENT v10.0 (Unified) ---")
        while True:
            prompt = "\nYou (Teaching): " if self.sys_mode["state"] == "TEACHING" else "\nYou: "
            try:
                u_in = input(prompt).strip()
            except EOFError:
                break
            if not u_in: continue
            if not self.run_turn(u_in): break

if __name__ == '__main__':
    bot = PipelineBot()
    bot.start()