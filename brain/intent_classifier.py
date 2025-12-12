import pickle
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from config import INTENT_MODEL_FILE

class IntentClassifier:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model = self._load()
        if not self.model: self.train_baseline()

    def _load(self):
        if os.path.exists(INTENT_MODEL_FILE):
            try:
                with open(INTENT_MODEL_FILE, 'rb') as f: return pickle.load(f)
            except: pass
        return None

    def train_baseline(self):
        print("BRAIN: Training robust Intent Classifier (Synthetic generation)...")
        
        # --- Synthetic Data Generator ---
        # We generate thousands of examples using templates to ensure robustness.
        
        entities = ["sun", "moon", "gravity", "python", "code", "life", "love", "ai", "robots", "space", "time", "history", "math", "logic", "biology", "cats", "dogs", "food", "water", "earth"]
        names = ["bob", "alice", "john", "jane", "sam", "alex", "charlie", "max", "user", "human"]
        foods = ["pizza", "pasta", "sushi", "burgers", "tacos", "salad", "steak", "curry", "ice cream", "cake"]
        adjectives = ["good", "bad", "cool", "weird", "hard", "easy", "fun", "boring", "red", "blue"]

        def generate_greeting():
            bases = ["hi", "hello", "hey", "sup", "yo", "greetings", "good morning", "good evening", "howdy"]
            modifiers = ["", " there", " bot", " friend", " buddy", " computer", " assistant"]
            punctuations = ["", ".", "!", "?", "!!"]
            return [f"{b}{m}{p}" for b in bases for m in modifiers for p in punctuations]

        def generate_farewell():
            bases = ["bye", "goodbye", "quit", "exit", "leave", "stop", "see ya", "cya", "later", "goodnight"]
            return [f"{b}{p}" for b in bases for p in ["", ".", "!"]]

        def generate_fact_teach():
            examples = []
            # "My X is Y" pattern
            for n in names: examples.append(f"my name is {n}")
            for f in foods: examples.append(f"i like {f}")
            for f in foods: examples.append(f"my favorite food is {f}")
            for a in adjectives: examples.append(f"i hate {a} things")
            # Contrastive / Complex
            examples.extend(["call me master", "i am a programmer", "i live in london", "remember that sky is blue", "note that birds fly"])
            return examples * 15 # Boost count

        def generate_qa_search():
            starts = ["what is", "tell me about", "how does", "explain", "define", "who is", "why is", "meaning of"]
            examples = []
            for s in starts:
                for e in entities:
                    examples.append(f"{s} {e}")
                    examples.append(f"{s} the {e}")
            # Complex queries
            examples.extend(["how do planes fly", "why is the sky blue", "capital of france", "distance to mars"])
            return examples

        def generate_fact_query():
            patterns = ["who am i", "what is my name", "do you know me", "what do i like", "recall my facts", "what is my favorite color"]
            return patterns * 20

        def generate_meta():
            patterns = ["debug", "show memory", "wipe data", "clear cache", "reset", "system status", "run self test", "version", "help", "capabilities"]
            return patterns * 20

        # Dataset Construction
        data = {
            "GREETING": generate_greeting(),
            "FAREWELL": generate_farewell(),
            "FACT_TEACH": generate_fact_teach(),
            "QA_SEARCH": generate_qa_search(),
            "FACT_QUERY": generate_fact_query(),
            "META": generate_meta()
        }

        X, y = [], []
        print(f"BRAIN: Dataset stats:")
        for label, phrases in data.items():
            # Deduplicate and shuffle
            phrases = list(set(phrases))
            print(f"  - {label}: {len(phrases)} examples")
            X.extend(phrases)
            y.extend([label]*len(phrases))
        
        # --- Model Training with Calibration ---
        print("BRAIN: Encoding vectors...")
        vecs = self.encoder.model.encode(X)
        
        print("BRAIN: Fitting Calibrated Classifier...")
        # 1. Class Weights: 'balanced' handles the uneven counts (e.g. fewer META vs QA)
        base_clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
        
        # 2. Calibration: Sigmoid (Platt Scaling) provides better probability estimates than raw LR
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
        
        calibrated_clf.fit(vecs, y)
        self.model = calibrated_clf
        
        with open(INTENT_MODEL_FILE, 'wb') as f: pickle.dump(self.model, f)
        print("BRAIN: Intent Model Saved.")

    def predict(self, vector):
        if not self.model: return "UNKNOWN", 0.0
        vector = vector.reshape(1, -1)
        # Calibrated classifier returns calibrated probabilities
        pred = self.model.predict(vector)[0]
        probs = self.model.predict_proba(vector)[0]
        return pred, max(probs)