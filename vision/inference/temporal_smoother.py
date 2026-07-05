from collections import Counter
from typing import List, Any, Optional

class TemporalSmoother:
    """
    Maintains a sliding window of predictions and returns the majority vote.
    Used to stabilize flickering predictions.
    """
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history: List[Any] = []
    
    def add_prediction(self, prediction: Any):
        """Adds a new prediction to the window."""
        self.history.append(prediction)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_stable_prediction(self) -> Optional[Any]:
        """Returns the most frequent prediction in the window."""
        if not self.history:
            return None
        
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]

    def clear(self):
        """Resets the history."""
        self.history = []

class PairedSmoother:
    """
    Stabilizes both label and confidence for a prediction task.
    """
    def __init__(self, window_size: int = 15):
        self.label_smoother = TemporalSmoother(window_size)
        self.confidence_history: List[float] = []
        self.window_size = window_size
        
    def add(self, label: str, confidence: float):
        self.label_smoother.add_prediction(label)
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.window_size:
            self.confidence_history.pop(0)
            
    def get(self) -> tuple[Optional[str], float]:
        label = self.label_smoother.get_stable_prediction()
        if not self.confidence_history:
            return label, 0.0
        
        avg_conf = sum(self.confidence_history) / len(self.confidence_history)
        return label, avg_conf

    def clear(self):
        self.label_smoother.clear()
        self.confidence_history = []
