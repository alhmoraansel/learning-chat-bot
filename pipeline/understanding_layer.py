from structures import PipelineState
from brain.service import NeuralBrainService
from config import CONFIDENCE_THRESHOLDS

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