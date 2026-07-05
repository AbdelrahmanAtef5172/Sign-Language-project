import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

@dataclass
class SessionState:
    """
    Holds per-session metadata and component outputs.
    Maintains state throughout the WebSocket connection lifecycle.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Vision Component Outputs
    gender: str = "male"
    emotion: str = "neutral"
    gender_conf: float = 0.0
    emotion_conf: float = 0.0
    
    # SLR Component Outputs
    slr_token_buffer: List[str] = field(default_factory=list)
    slr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # LLM Component Outputs
    llm_context: List[str] = field(default_factory=list)
    llm_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Session Metadata
    frame_count: int = 0
    last_audio_timestamp: float = 0.0
    
    def reset_smoother_state(self):
        """Resets fields related to temporal smoothing when face is lost."""
        self.gender_conf = 0.0
        self.emotion_conf = 0.0
        # Note: Actual smoother history is handled inside VisionInferenceEngine
