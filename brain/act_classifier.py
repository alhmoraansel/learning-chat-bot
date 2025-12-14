import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from config import ACT_MODEL_FILE

class ActClassifier:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model = self._load()
        if not self.model: self.train_baseline()

    def _load(self):
        if os.path.exists(ACT_MODEL_FILE):
            try:
                with open(ACT_MODEL_FILE, 'rb') as f: return pickle.load(f)
            except: pass
        return None

    def train_baseline(self):
        print("BRAIN: Training robust Act Classifier (Synthetic generation)...")
        
        def gen_affirm():
            bases = ["yes", "yeah", "yep", "correct", "right", "true", "sure", "ok", "okay", "fine", "indeed", "absolutely", "definitely"]
            return [f"{b}{p}" for b in bases for p in ["", ".", "!"]]

        def gen_deny():
            bases = ["no", "nope", "nah", "wrong", "false", "incorrect", "negative", "not really", "never", "invalid"]
            phrases = ["that is wrong", "you are wrong", "not true", "incorrect answer"]
            return [f"{b}{p}" for b in bases for p in ["", ".", "!"]] + phrases

        def gen_request():
            starts = ["tell me", "what is", "explain", "describe", "how to", "show me", "define", "can you", "could you"]
            topics = ["this", "that", "it", "gravity", "life", "python"]
            return [f"{s} {t}" for s in starts for t in topics]

        def gen_give_info():
            starts = ["it is", "the answer is", "because", "basically", "usually", "i think", "i believe"]
            return [f"{s} something" for s in starts] + ["this is a fact", "facts are useful"]

        def gen_clarify():
            return ["what?", "huh?", "pardon?", "excuse me?", "what do you mean?", "can you repeat?", "i dont understand", "clarify please"]

        def gen_followup():
            connectors = ["and", "but", "so", "then", "also", "plus"]
            queries = ["what about x", "how big is it", "why", "when"]
            return [f"{c} {q}" for c in connectors for q in queries]

        def gen_emotion():
            words = ["awesome", "cool", "great", "wow", "amazing", "nice", "perfect", "excellent", "brilliant", "wonderful", "thanks", "thank you"]
            return [f"{w}{p}" for w in words for p in ["", "!", "!!"]]

        data = {
            "AFFIRM": gen_affirm(),
            "DENY": gen_deny(),
            "REQUEST_INFO": gen_request(),
            "GIVE_INFO": gen_give_info(),
            "CLARIFY": gen_clarify(),
            "FOLLOWUP": gen_followup(),
            "EMOTION": gen_emotion() # New Class
        }

        X, y = [], []
        for label, phrases in data.items():
            phrases = list(set(phrases))
            X.extend(phrases)
            y.extend([label]*len(phrases))
        
        vecs = self.encoder.model.encode(X)
        
        base_clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
        
        calibrated_clf.fit(vecs, y)
        self.model = calibrated_clf
        
        with open(ACT_MODEL_FILE, 'wb') as f: pickle.dump(self.model, f)
        print("BRAIN: Act Model Saved.")

    def predict(self, vector):
        if not self.model: return "UNKNOWN", 0.0
        vector = vector.reshape(1, -1)
        pred = self.model.predict(vector)[0]
        probs = self.model.predict_proba(vector)[0]
        return pred, max(probs)