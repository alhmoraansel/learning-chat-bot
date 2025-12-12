from transformers import pipeline
from config import ENABLE_GENERATIVE, GENERATIVE_MODEL_NAME

class Generator:
    def __init__(self):
        self.model = None
        if ENABLE_GENERATIVE:
            try:
                print(f"BRAIN: Loading GenAI ({GENERATIVE_MODEL_NAME})...")
                self.model = pipeline('text-generation', model=GENERATIVE_MODEL_NAME)
            except Exception as e:
                print(f"BRAIN: GenAI load failed: {e}")

    def generate(self, context_str):
        if not self.model: return None
        try:
            prompt = f"Question: {context_str}\nAnswer:"
            output = self.model(prompt, max_new_tokens=30, num_return_sequences=1, do_sample=True, temperature=0.7)
            return output[0]['generated_text'].replace(prompt, "").strip()
        except: return None