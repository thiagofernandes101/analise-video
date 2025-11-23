"""
Person tracking model representing a tracked person with pose information.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from models.bounding_box import BoundingBox


@dataclass
class PersonTracking:
    """
    Represents a tracked person in the video with pose keypoints.
    
    Attributes:
        track_id: Unique identifier for this person across frames
        bounding_box: Bounding box around the person
        keypoints: Array of keypoint coordinates (17, 2) for COCO format
        confidence: Detection confidence score (0.0 to 1.0)
    """
    track_id: int
    bounding_box: BoundingBox
    keypoints: np.ndarray  # Shape: (17, 2)
    confidence: float = 1.0
    
    def get_keypoint_safely(self, index: int) -> np.ndarray:
        """
        Safely retrieve a keypoint by index.
        
        Args:
            index: Keypoint index (0-16 for COCO format)
            
        Returns:
            Keypoint coordinates as [x, y] array, or [0, 0] if invalid
        """
        if 0 <= index < len(self.keypoints):
            return self.keypoints[index]
        return np.array([0, 0])
    
    def is_keypoint_visible(self, index: int, threshold: float = 1.0) -> bool:
        """
        Check if a keypoint is visible (coordinates > threshold).
        
        Args:
            index: Keypoint index
            threshold: Minimum coordinate value to consider visible
            
        Returns:
            True if keypoint is visible, False otherwise
        """
        keypoint = self.get_keypoint_safely(index)
        return keypoint[0] > threshold and keypoint[1] > threshold
