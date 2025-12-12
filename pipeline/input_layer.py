import re
import time
from structures import PipelineState

class InputLayer:
    @staticmethod
    def process(state: PipelineState) -> PipelineState:
        text = state.raw_text
        if len(text) > 500: 
            state.response = "Input too long."
            state.stop_pipeline = True
            return state
        norm = re.sub(r'[^\w\s]', '', text.lower().strip())
        state.norm_text = norm
        state.timestamp = time.time()
        return state