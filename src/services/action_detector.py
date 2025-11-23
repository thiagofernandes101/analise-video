"""
Action detection service.

Responsible for detecting dynamic actions using temporal keypoint data.
Single Responsibility: Temporal action classification.
"""
import numpy as np
from typing import List, Optional

from models.person_tracking import PersonTracking
from services.keypoint_history_manager import KeypointHistoryManager
from config import Config


class ActionDetector:
    """
    Detects dynamic actions from temporal keypoint data.
    
    Uses keypoint history to identify movements like jumping, walking, waving.
    """
    
    def __init__(
        self, 
        history_manager: KeypointHistoryManager,
        config: Config = Config()
    ):
        """
        Initialize action detector.
        
        Args:
            history_manager: History manager for temporal data
            config: Configuration object
        """
        self._history_manager = history_manager
        self._config = config
    
    def detect_action(
        self, 
        person: PersonTracking, 
        current_posture: str
    ) -> Optional[str]:
        """
        Detect dynamic action for a person.
        
        Args:
            person: Current person tracking data
            current_posture: Current posture classification
            
        Returns:
            Action label if detected, None otherwise
        """
        if not self._history_manager.has_sufficient_history(person.track_id, 6):
            return self._detect_static_action(person)
        
        # Try detecting various actions in priority order
        action = self._detect_jumping(person)
        if action:
            return action
        
        action = self._detect_walking(person, current_posture)
        if action:
            return action
        
        action = self._detect_hand_waving(person)
        if action:
            return action
        
        action = self._detect_head_waving(person)
        if action:
            return action
        
        # Fall back to static actions
        return self._detect_static_action(person)
    
    def _detect_static_action(self, person: PersonTracking) -> Optional[str]:
        """
        Detect static arm actions (hands up).
        
        Args:
            person: Person tracking data
            
        Returns:
            Action label if hands are up, None otherwise
        """
        left_wrist = person.get_keypoint_safely(9)
        right_wrist = person.get_keypoint_safely(10)
        left_shoulder = person.get_keypoint_safely(5)
        right_shoulder = person.get_keypoint_safely(6)
        
        left_hand_up = (
            left_wrist[1] > 0 and left_shoulder[1] > 0 and 
            left_wrist[1] < left_shoulder[1]
        )
        
        right_hand_up = (
            right_wrist[1] > 0 and right_shoulder[1] > 0 and 
            right_wrist[1] < right_shoulder[1]
        )
        
        if left_hand_up or right_hand_up:
            return "Hands Up"
        
        return None
    
    def _detect_jumping(self, person: PersonTracking) -> Optional[str]:
        """Detect jumping based on vertical hip movement."""
        history = self._history_manager.get_recent_history(person.track_id, 10)
        
        if len(history) < 3:
            return None
        
        recent_hip_y_positions = [
            (keypoints[11][1] + keypoints[12][1]) / 2 
            for keypoints in history
        ]
        
        y_min = min(recent_hip_y_positions)
        y_max = max(recent_hip_y_positions)
        y_range = y_max - y_min
        
        # Get current hip position
        current_hip_y = (
            person.get_keypoint_safely(11)[1] + 
            person.get_keypoint_safely(12)[1]
        ) / 2
        
        current_shoulder_y = (
            person.get_keypoint_safely(5)[1] + 
            person.get_keypoint_safely(6)[1]
        ) / 2
        
        person_height = abs(current_hip_y - current_shoulder_y) * 2
        
        if person_height <= 0:
            return None
        
        height_ratio_threshold = self._config.activity.JUMPING_HEIGHT_RATIO
        if (y_range / person_height) <= height_ratio_threshold:
            return None
        
        # Check if currently in air
        average_y = np.mean(recent_hip_y_positions)
        air_threshold = person_height * self._config.activity.JUMPING_AIR_RATIO
        
        if current_hip_y < (average_y - air_threshold):
            return "Jumping"
        
        return None
    
    def _detect_walking(
        self, 
        person: PersonTracking, 
        current_posture: str
    ) -> Optional[str]:
        """Detect walking based on ankle movement variance."""
        if current_posture != "Standing":
            return None
        
        history = self._history_manager.get_recent_history(person.track_id, 15)
        
        if len(history) < 5:
            return None
        
        left_ankle_x_positions = [kps[15][0] for kps in history]
        right_ankle_x_positions = [kps[16][0] for kps in history]
        
        left_variance = np.var(left_ankle_x_positions)
        right_variance = np.var(right_ankle_x_positions)
        
        threshold = self._config.activity.WALKING_ANKLE_VARIANCE
        
        if left_variance > threshold or right_variance > threshold:
            return "Walking"
        
        return None
    
    def _detect_hand_waving(self, person: PersonTracking) -> Optional[str]:
        """Detect hand waving based on wrist movement."""
        history = self._history_manager.get_recent_history(person.track_id, 10)
        
        if len(history) < 5:
            return None
        
        left_wrist = person.get_keypoint_safely(9)
        left_shoulder = person.get_keypoint_safely(5)
        right_wrist = person.get_keypoint_safely(10)
        right_shoulder = person.get_keypoint_safely(6)
        
        # Check left hand waving
        if left_wrist[1] < left_shoulder[1] and left_wrist[0] > 0:
            left_wrist_x_positions = [kps[9][0] for kps in history]
            if np.var(left_wrist_x_positions) > self._config.activity.WAVING_HAND_VARIANCE:
                return "Waving Hand"
        
        # Check right hand waving
        if right_wrist[1] < right_shoulder[1] and right_wrist[0] > 0:
            right_wrist_x_positions = [kps[10][0] for kps in history]
            if np.var(right_wrist_x_positions) > self._config.activity.WAVING_HAND_VARIANCE:
                return "Waving Hand"
        
        return None
    
    def _detect_head_waving(self, person: PersonTracking) -> Optional[str]:
        """Detect head waving/bobbing based on nose movement."""
        history = self._history_manager.get_recent_history(person.track_id, 10)
        
        if len(history) < 5:
            return None
        
        nose_x_positions = [kps[0][0] for kps in history]
        hip_x_positions = [(kps[11][0] + kps[12][0]) / 2 for kps in history]
        
        nose_variance = np.var(nose_x_positions)
        hip_variance = np.var(hip_x_positions)
        
        # Head moving but body stable
        if (nose_variance > self._config.activity.WAVING_HEAD_VARIANCE and 
            hip_variance < self._config.activity.HEAD_MOVEMENT_BODY_STABILITY):
            return "Waving Head"
        
        return None
