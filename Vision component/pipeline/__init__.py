"""
Pipeline stages for the gender detection vision component.
"""

from .component import GenderDetectionComponent
from .frame_gate import FrameGate
from .face_detector import FaceDetector
from .face_aligner import FaceAligner
from .gender_classifier import GenderInference
from .smoother import ResultSmoother
