from pydantic import BaseModel
from typing import Literal, Optional, List, Union

class ControlMessage(BaseModel):
    """JSON message from client to server for session control."""
    type: Literal["control"] = "control"
    action: Literal["pause", "resume", "disconnect"]

class StatusMessage(BaseModel):
    """JSON message from server to client with pipeline status."""
    type: Literal["status"] = "status"
    gender: str
    emotion: str
    recognized_text: str

class ErrorMessage(BaseModel):
    """JSON message from server to client on component failure."""
    type: Literal["error"] = "error"
    component: str
    message: str

# Note: Video frames and Audio chunks are sent as raw binary messages,
# so they don't have Pydantic models.
