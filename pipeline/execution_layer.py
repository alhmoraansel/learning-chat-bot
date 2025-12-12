import time
from sklearn.metrics.pairwise import cosine_similarity
from structures import PipelineState, TopicNode, WorkingMemory
from memory.storage import StorageService
from brain.service import NeuralBrainService
from utils import Personality

class ExecutionLayer:
    def __init__(self, db: StorageService, memory: WorkingMemory, sys_mode: dict, brain: NeuralBrainService):
        self.db = db
        self.mem = memory
        self.mode = sys_mode
        self.brain = brain

    def process(self, state: PipelineState) -> PipelineState:
        # Confirming Fact State
        if self.mode["state"] == "CONFIRMING_FACT":
            if state.norm_text in ["yes", "yeah", "sure", "ok"]:
                self.db.save_fact(self.mem.pending_fact['key'], self.mem.pending_fact['val'])
                state.response = "Okay, updated your profile."
            else:
                state.response = "Got it, keeping the old setting."
            self.mode["state"] = "IDLE"
            self.mem.pending_fact = None
            return state

        # Teaching State
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
                if self.brain.generator:
                     gen_text = self.brain.generator.generate(txt)
                     if gen_text: txt = gen_text
                
                formatted = Personality.format(style, txt)
                final_responses.append(formatted)
                
                if state.query_vector is not None:
                    self.mem.last_memory_vector = state.query_vector

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
                if target_vec is not None and cache['vectors'] is not None:
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