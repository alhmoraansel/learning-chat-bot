from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL_NAME

class Encoder:
    def __init__(self):
        print(f"BRAIN: Loading Encoder ({EMBEDDING_MODEL_NAME})...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0].astype(np.float32)