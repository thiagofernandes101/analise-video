"""
Detector protocol interface.

Defines the contract for detection components (pose, face, etc.).
"""
from typing import Protocol, List, Any
import numpy as np


class DetectorProtocol(Protocol):
    """
    Protocol for detector components.
    
    Any detector implementation should provide these methods.
    """
    
    def detect(self, frame: np.ndarray) -> Any:
        """
        Perform detection on a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Detection results (format depends on detector type)
        """
        ...
    
    def warmup(self) -> None:
        """
        Perform warmup/initialization of the detector.
        
        This is typically used to load models and run dummy inference.
        """
        ...
