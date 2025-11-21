"""
Activity recognition result model.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActivityResult:
    """
    Represents the result of activity recognition.
    
    Attributes:
        posture: Detected posture (e.g., 'Standing', 'Sitting', 'Laying Down')
        action: Detected action (e.g., 'Walking', 'Jumping', 'Waving Hand')
        confidence: Overall confidence score (0.0 to 1.0)
    """
    posture: str
    action: Optional[str] = None
    confidence: float = 1.0
    
    def get_display_label(self) -> str:
        """
        Get a human-readable label for display.
        
        Returns:
            Formatted label combining posture and action
        """
        if self.action:
            return f"{self.posture}, {self.action}"
        return self.posture
    
    @classmethod
    def unknown(cls) -> 'ActivityResult':
        """Create an unknown activity result."""
        return cls(posture="Unknown", action=None, confidence=0.0)
