from structures import WorkingMemory, PipelineState
from utils import BackupManager
from config import ENABLE_DEBUG_THOUGHTS
from brain.service import NeuralBrainService
from memory.storage import StorageService
from pipeline.input_layer import InputLayer
from pipeline.perception_layer import PerceptionLayer
from pipeline.understanding_layer import UnderstandingLayer
from pipeline.context_layer import ContextLayer
from pipeline.reasoning_layer import ReasoningLayer
from pipeline.retrieval_layer import RetrievalLayer
from pipeline.execution_layer import ExecutionLayer

class PipelineBot:
    def __init__(self):
        BackupManager.create_backup()
        
        self.brain = NeuralBrainService()
        self.db = StorageService()
        self.memory = WorkingMemory()
        self.sys_mode = {"state": "IDLE"}
        
        self.layers = [
            InputLayer(),
            PerceptionLayer(self.brain),
            UnderstandingLayer(self.brain),
            ContextLayer(self.memory, self.db, self.brain),
            ReasoningLayer(),
            RetrievalLayer(self.brain, self.db),
            ExecutionLayer(self.db, self.memory, self.sys_mode, self.brain)
        ]

    def run_turn(self, user_input: str) -> bool:
        state = PipelineState(raw_text=user_input)
        
        for layer in self.layers:
            state = layer.process(state)
            if state.stop_pipeline: break
        
        if ENABLE_DEBUG_THOUGHTS and state.thought_trace:
            print(f"\n[THOUGHTS]: {state.thought_trace}")
            
        if state.response:
            print(f"Bot: {state.response}")
            
        return not state.stop_pipeline

    def start(self):
        print("\n--- NEURAL AGENT v10.0 (Modular) ---")
        while True:
            prompt = "\nYou (Teaching): " if self.sys_mode["state"] == "TEACHING" else "\nYou: "
            u_in = input(prompt).strip()
            if not u_in: continue
            if not self.run_turn(u_in): break

if __name__ == '__main__':
    bot = PipelineBot()
    bot.start()