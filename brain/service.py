import pickle
import os
import numpy as np
from sklearn.decomposition import PCA
from config import PCA_MODEL_FILE, TOPIC_MODEL_FILE
from brain.encoder import Encoder
from brain.intent_classifier import IntentClassifier
from brain.act_classifier import ActClassifier
from brain.generator import Generator

class NeuralBrainService:
    def __init__(self):
        self.encoder = Encoder()
        self.intent_classifier = IntentClassifier(self.encoder)
        self.act_classifier = ActClassifier(self.encoder)
        self.generator = Generator()
        
        self.pca = self._load_pickle(PCA_MODEL_FILE)
        self.topic_model = self._load_pickle(TOPIC_MODEL_FILE)
        
        # Cache
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
        if self.pca and len(vectors) > 0:
            if len(vectors.shape) == 1: vectors = vectors.reshape(1, -1)
            return self.pca.transform(vectors).astype(np.float32)
        return vectors

    def train_pca(self, vectors):
        if len(vectors) < 50: return
        self.pca = PCA(n_components=min(128, len(vectors), vectors.shape[1]))
        self.pca.fit(vectors)
        with open(PCA_MODEL_FILE, 'wb') as f: pickle.dump(self.pca, f)
        self.is_cache_dirty = True