"""
Face detection model representing a detected face in an image.
"""
from dataclasses import dataclass
from typing import Optional

from models.bounding_box import BoundingBox


@dataclass
class FaceDetection:
    """
    Represents a detected face in the image.
    
    Attributes:
        bounding_box: Bounding box around the face
        confidence: Detection confidence score (0.0 to 1.0)
        person_id: Optional ID of the person this face belongs to
    """
    bounding_box: BoundingBox
    confidence: float = 1.0
    person_id: Optional[int] = None
    
    @property
    def has_person_assignment(self) -> bool:
        """Check if this face has been assigned to a person."""
        return self.person_id is not None
