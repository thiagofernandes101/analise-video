"""
Emotion analysis result model.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionResult:
    """
    Represents the result of emotion analysis.
    
    Attributes:
        emotion: Detected emotion name (e.g., 'happy', 'sad', 'neutral')
        confidence: Confidence score for this emotion (0.0 to 100.0)
        frame_number: Frame number when this was analyzed
    """
    emotion: str
    confidence: float
    frame_number: int
    
    def is_confident(self, threshold: float) -> bool:
        """
        Check if emotion detection confidence meets the threshold.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            True if confidence >= threshold
        """
        return self.confidence >= threshold
    
    @classmethod
    def unknown(cls, frame_number: int) -> 'EmotionResult':
        """Create an unknown emotion result."""
        return cls(emotion="Unknown", confidence=0.0, frame_number=frame_number)
    
    @classmethod
    def analyzing(cls, frame_number: int) -> 'EmotionResult':
        """Create a result indicating analysis in progress."""
        return cls(emotion="Analyzing...", confidence=0.0, frame_number=frame_number)
