from structures import PipelineState
from brain.service import NeuralBrainService

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