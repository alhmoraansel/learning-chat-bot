import time
from structures import PipelineState, UserProfile, WorkingMemory
from memory.storage import StorageService
from brain.service import NeuralBrainService
from config import CONTEXT_TIMEOUT_SECONDS

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