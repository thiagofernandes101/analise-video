"""
Keypoint history manager service.

Manages temporal storage of keypoints for activity recognition.
Single Responsibility: Store and retrieve keypoint history.
"""
from collections import deque
from typing import Dict, List
import numpy as np


class KeypointHistoryManager:
    """
    Manages historical keypoint data for tracked persons.
    
    This class has a single responsibility: maintain a sliding window
    of keypoint data for each tracked person to enable temporal analysis.
    """
    
    def __init__(self, history_length: int = 30):
        """
        Initialize the history manager.
        
        Args:
            history_length: Maximum number of frames to keep in history
        """
        self._history: Dict[int, deque] = {}
        self._history_length = history_length
    
    def update_history(self, person_id: int, keypoints: np.ndarray) -> None:
        """
        Add keypoints for a person to their history.
        
        Args:
            person_id: Unique identifier for the person
            keypoints: Keypoint array to store
        """
        if person_id not in self._history:
            self._history[person_id] = deque(maxlen=self._history_length)
        
        self._history[person_id].append(keypoints)
    
    def get_history(self, person_id: int) -> List[np.ndarray]:
        """
        Get keypoint history for a person.
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            List of keypoint arrays (oldest to newest)
        """
        if person_id not in self._history:
            return []
        
        return list(self._history[person_id])
    
    def get_recent_history(self, person_id: int, num_frames: int) -> List[np.ndarray]:
        """
        Get the most recent N frames of history.
        
        Args:
            person_id: Unique identifier for the person
            num_frames: Number of recent frames to retrieve
            
        Returns:
            List of recent keypoint arrays
        """
        history = self.get_history(person_id)
        return history[-num_frames:] if history else []
    
    def has_sufficient_history(self, person_id: int, minimum_frames: int) -> bool:
        """
        Check if person has enough recorded history.
        
        Args:
            person_id: Unique identifier for the person
            minimum_frames: Minimum number of frames required
            
        Returns:
            True if history length >= minimum_frames
        """
        return len(self.get_history(person_id)) >= minimum_frames
    
    def clear_person_history(self, person_id: int) -> None:
        """
        Clear history for a specific person.
        
        Args:
            person_id: Unique identifier for the person
        """
        if person_id in self._history:
            del self._history[person_id]
    
    def cleanup_inactive_persons(self, active_person_ids: List[int]) -> None:
        """
        Remove history for persons no longer being tracked.
        
        Args:
            active_person_ids: List of currently active person IDs
        """
        inactive_ids = [
            person_id for person_id in self._history.keys()
            if person_id not in active_person_ids
        ]
        
        for person_id in inactive_ids:
            self.clear_person_history(person_id)
