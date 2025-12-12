from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class TopicNode:
    id: int
    centroid: np.ndarray
    last_active: float
    label: str = "general"

@dataclass
class UserProfile:
    facts: Dict[str, str] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None 

@dataclass
class WorkingMemory:
    active_topic: Optional[TopicNode] = None
    last_turn_vector: Optional[np.ndarray] = None
    last_memory_vector: Optional[np.ndarray] = None 
    pending_correction: Optional[Dict] = None
    pending_fact: Optional[Dict] = None 
    history_window: List[Dict] = field(default_factory=list)

@dataclass
class PipelineState:
    raw_text: str
    norm_text: str = ""
    timestamp: float = 0.0
    
    # Perception
    raw_vector: Optional[np.ndarray] = None
    features: Dict[str, bool] = field(default_factory=dict)
    
    # Understanding
    intent: str = "UNKNOWN"
    intent_conf: float = 0.0
    act: str = "UNKNOWN"
    act_conf: float = 0.0
    
    # Context & Profile
    query_vector: Optional[np.ndarray] = None 
    weighted_context: Optional[np.ndarray] = None
    user_profile: Optional[UserProfile] = None
    
    # Reasoning
    plan: List[Dict[str, Any]] = field(default_factory=list)
    thought_trace: List[str] = field(default_factory=list)
    
    # Execution
    response: Optional[str] = None
    stop_pipeline: bool = False