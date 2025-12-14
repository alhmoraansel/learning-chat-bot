import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from structures import PipelineState
from brain.service import NeuralBrainService
from memory.storage import StorageService
from config import CONFIDENCE_THRESHOLDS, W_SEMANTIC, W_CONTEXT, W_RECENCY, W_PROFILE

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
                if cache['vectors'] is None or len(cache['vectors']) == 0:
                    new_plan.append({"op": "FALLBACK_TEACH"})
                    continue
                
                q_vec = step.get("query_vector")
                if q_vec is None and "query_text" in step:
                    q_vec = self.brain.get_embedding(step["query_text"])
                
                q_pca = self.brain.apply_pca(q_vec).reshape(1, -1)
                
                # Semantic
                sem_sims = cosine_similarity(q_pca, cache['vectors'])[0]
                
                # Context
                ctx_sims = np.zeros_like(sem_sims)
                if state.weighted_context is not None:
                    c_pca = self.brain.apply_pca(state.weighted_context).reshape(1, -1)
                    ctx_sims = cosine_similarity(c_pca, cache['vectors'])[0]
                
                # Profile
                prof_sims = np.zeros_like(sem_sims)
                if state.user_profile and state.user_profile.embedding is not None:
                    p_pca = self.brain.apply_pca(state.user_profile.embedding).reshape(1, -1)
                    prof_sims = cosine_similarity(p_pca, cache['vectors'])[0]

                # Recency
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
                    new_plan.append({"op": "FALLBACK_TEACH"})
            
            elif step["op"] == "SYNTHESIZE":
                if retrieved_results:
                    combined = " ".join(retrieved_results)
                    new_plan.append({"op": "RESPOND", "text": f"Combined: {combined}"})
                else:
                     new_plan.append({"op": "FALLBACK_TEACH"})
            else:
                new_plan.append(step)
        
        state.plan = new_plan
        return state