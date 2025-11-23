"""
Posture detection service.

Responsible for detecting static postures: standing, sitting, laying down.
Single Responsibility: Posture classification based on body angles and proportions.
"""
import numpy as np
from typing import Tuple

from models.person_tracking import PersonTracking
from config import Config


class PostureDetector:
    """
    Detects static postures from keypoint data.
    
    Analyzes body angles and segment proportions to classify posture.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize posture detector.
        
        Args:
            config: Configuration object
        """
        self._config = config
    
    def detect_posture(self, person: PersonTracking) -> str:
        """
        Detect the current posture of a person.
        
        Args:
            person: PersonTracking data with keypoints
            
        Returns:
            Posture label: 'Standing', 'Sitting', 'Laying Down', or 'Unknown'
        """
        if not self._has_required_keypoints(person):
            return "Unknown"
        
        shoulder_center = self._get_shoulder_center(person)
        hip_center = self._get_hip_center(person)
        
        if not self._are_points_valid(shoulder_center, hip_center):
            return "Standing"  # Default fallback
        
        torso_angle = self._calculate_torso_angle(shoulder_center, hip_center)
        
        if self._is_laying_down(torso_angle):
            return "Laying Down"
        
        return self._classify_sitting_or_standing(person, shoulder_center, hip_center)
    
    def _has_required_keypoints(self, person: PersonTracking) -> bool:
        """Check if person has minimum required keypoints visible."""
        required_indices = [5, 6, 11, 12]  # Shoulders and hips
        return all(person.is_keypoint_visible(idx) for idx in required_indices)
    
    def _get_shoulder_center(self, person: PersonTracking) -> np.ndarray:
        """Calculate center point between shoulders."""
        left_shoulder = person.get_keypoint_safely(5)
        right_shoulder = person.get_keypoint_safely(6)
        return (left_shoulder + right_shoulder) / 2
    
    def _get_hip_center(self, person: PersonTracking) -> np.ndarray:
        """Calculate center point between hips."""
        left_hip = person.get_keypoint_safely(11)
        right_hip = person.get_keypoint_safely(12)
        return (left_hip + right_hip) / 2
    
    def _are_points_valid(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """Check if two points have valid coordinates."""
        return point1[0] > 0 and point1[1] > 0 and point2[0] > 0 and point2[1] > 0
    
    def _calculate_torso_angle(
        self, 
        shoulder_center: np.ndarray, 
        hip_center: np.ndarray
    ) -> float:
        """
        Calculate the angle of the torso from vertical.
        
        Args:
            shoulder_center: Center point between shoulders
            hip_center: Center point between hips
            
        Returns:
            Angle in degrees from vertical axis
        """
        delta_y = hip_center[1] - shoulder_center[1]
        delta_x = hip_center[0] - shoulder_center[0]
        angle = np.degrees(np.arctan2(abs(delta_x), abs(delta_y)))
        return angle
    
    def _is_laying_down(self, torso_angle: float) -> bool:
        """
        Check if torso angle indicates laying down posture.
        
        Args:
            torso_angle: Angle from vertical in degrees
            
        Returns:
            True if laying down
        """
        threshold = self._config.activity.LAYING_DOWN_ANGLE_THRESHOLD
        return torso_angle > threshold
    
    def _classify_sitting_or_standing(
        self,
        person: PersonTracking,
        shoulder_center: np.ndarray,
        hip_center: np.ndarray
    ) -> str:
        """
        Classify between sitting and standing based on leg proportions.
        
        Args:
            person: PersonTracking data
            shoulder_center: Shoulder center point
            hip_center: Hip center point
            
        Returns:
            'Sitting' or 'Standing'
        """
        left_knee = person.get_keypoint_safely(13)
        right_knee = person.get_keypoint_safely(14)
        
        if not (person.is_keypoint_visible(13) and person.is_keypoint_visible(14)):
            return "Standing"  # Default if knees not visible
        
        knee_y = (left_knee[1] + right_knee[1]) / 2
        hip_y = hip_center[1]
        
        if knee_y <= 0 or hip_y <= 0:
            return "Standing"
        
        thigh_vertical_length = abs(knee_y - hip_y)
        torso_length = abs(hip_y - shoulder_center[1])
        
        if torso_length <= 0:
            return "Standing"
        
        thigh_to_torso_ratio = thigh_vertical_length / torso_length
        threshold = self._config.activity.SITTING_THIGH_RATIO_THRESHOLD
        
        if thigh_to_torso_ratio < threshold:
            return "Sitting"  # Short thigh projection = sitting
        else:
            return "Standing"  # Long thigh projection = standing
