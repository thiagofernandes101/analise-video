"""
Statistics tracker service.

Responsible for collecting and aggregating video analysis statistics during
processing. Implements anomaly detection and maintains per-person tracking.

Single Responsibility: Statistics collection and anomaly detection.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import deque

from models.person_tracking import PersonTracking
from models.activity_result import ActivityResult
from models.emotion_result import EmotionResult
from models.video_statistics import VideoStatistics, PersonStatistics, AnomalyEvent
from config import Config


class StatisticsTracker:
    """
    Tracks comprehensive video analysis statistics.
    
    Collects data throughout video processing including per-person tracking,
    anomaly detection, and global activity/emotion distributions.
    
    Follows Single Responsibility Principle: Only handles statistics tracking.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize statistics tracker.
        
        Args:
            config: Configuration object
        """
        self._config = config
        self._statistics = VideoStatistics()
        
        # Tracking data for anomaly detection
        self._last_keypoints: Dict[int, np.ndarray] = {}
        self._activity_history: Dict[int, deque] = {}
        
        # Current frame number
        self._current_frame = 0
    
    def update(
        self,
        frame_number: int,
        persons: List[PersonTracking],
        activities: Dict[int, ActivityResult],
        emotions: Dict[int, EmotionResult]
    ) -> None:
        """
        Update statistics with data from current frame.
        
        Args:
            frame_number: Current frame number
            persons: List of detected persons
            activities: Activity results mapped by person ID
            emotions: Emotion results mapped by person ID
        """
        self._current_frame = frame_number
        self._statistics.total_frames = frame_number + 1
        
        for person in persons:
            track_id = person.track_id
            
            # Ensure person stats exist
            if track_id not in self._statistics.person_stats:
                self._statistics.person_stats[track_id] = PersonStatistics(track_id)
            
            person_stats = self._statistics.person_stats[track_id]
            person_stats.frame_count += 1
            
            # Update activity tracking
            if track_id in activities:
                activity = activities[track_id]
                activity_label = activity.get_display_label()
                
                # Add to person stats
                person_stats.add_activity(activity_label)
                
                # Update global distribution
                if activity_label not in self._statistics.activity_distribution:
                    self._statistics.activity_distribution[activity_label] = 0
                self._statistics.activity_distribution[activity_label] += 1
                
                # Check for anomalies
                anomaly = self._detect_anomaly(person, activity)
                if anomaly:
                    person_stats.add_anomaly(anomaly)
                    self._statistics.all_anomalies.append(anomaly)
            
            # Update emotion tracking
            if track_id in emotions:
                emotion = emotions[track_id]
                emotion_label = emotion.emotion
                
                # Add to person stats
                person_stats.add_emotion(emotion_label)
                
                # Update global distribution (only if confident)
                if emotion_label != "Unknown" and emotion_label != "Analyzing":
                    if emotion_label not in self._statistics.emotion_distribution:
                        self._statistics.emotion_distribution[emotion_label] = 0
                    self._statistics.emotion_distribution[emotion_label] += 1
    
    def _detect_anomaly(
        self,
        person: PersonTracking,
        activity: ActivityResult
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly for a person.
        
        Checks for:
        1. Abrupt movements (large keypoint changes)
        2. Atypical activity combinations
        3. Rapid activity transitions
        
        Args:
            person: Person tracking data
            activity: Current activity result
            
        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        track_id = person.track_id
        
        # Check 1: Abrupt movement
        if track_id in self._last_keypoints:
            movement = self._calculate_movement(
                self._last_keypoints[track_id],
                person.keypoints
            )
            
            threshold = self._config.summary.ANOMALY_MOVEMENT_THRESHOLD
            if movement > threshold:
                return AnomalyEvent(
                    frame_number=self._current_frame,
                    track_id=track_id,
                    anomaly_type="abrupt_movement",
                    explanation=f"Movimento abrupto: {movement:.0f} pixels em 1 frame",
                    activity=activity.get_display_label(),
                    posture=activity.posture
                )
        
        # Store current keypoints for next frame
        self._last_keypoints[track_id] = person.keypoints.copy()
        
        # Check 2: Atypical combinations
        atypical_anomaly = self._check_atypical_combination(track_id, activity)
        if atypical_anomaly:
            return atypical_anomaly
        
        # Check 3: Rapid transitions
        transition_anomaly = self._check_rapid_transitions(track_id, activity)
        if transition_anomaly:
            return transition_anomaly
        
        return None
    
    def _calculate_movement(
        self,
        prev_keypoints: np.ndarray,
        curr_keypoints: np.ndarray
    ) -> float:
        """
        Calculate total movement between keypoint frames.
        
        Args:
            prev_keypoints: Previous frame keypoints
            curr_keypoints: Current frame keypoints
            
        Returns:
            Total movement in pixels
        """
        # Calculate Euclidean distance for each keypoint
        movements = []
        for i in range(min(len(prev_keypoints), len(curr_keypoints))):
            prev_x, prev_y = prev_keypoints[i]
            curr_x, curr_y = curr_keypoints[i]
            
            # Skip if keypoint not detected (0, 0)
            if prev_x > 0 and prev_y > 0 and curr_x > 0 and curr_y > 0:
                distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                movements.append(distance)
        
        # Return average movement
        return np.mean(movements) if movements else 0.0
    
    def _check_atypical_combination(
        self,
        track_id: int,
        activity: ActivityResult
    ) -> Optional[AnomalyEvent]:
        """
        Check for atypical posture/action combinations.
        
        Args:
            track_id: Person ID
            activity: Current activity result
            
        Returns:
            AnomalyEvent if atypical combination detected
        """
        # Define atypical combinations
        atypical_combinations = {
            ("Sitting", "Walking"): "Sitting enquanto fazendo Walking",
            ("Sitting", "Jumping"): "Sitting enquanto fazendo Jumping",
            ("Laying Down", "Walking"): "Laying Down enquanto fazendo Walking",
            ("Laying Down", "Jumping"): "Laying Down enquanto fazendo Jumping",
        }
        
        posture = activity.posture
        action = activity.action
        
        if action and (posture, action) in atypical_combinations:
            explanation = atypical_combinations[(posture, action)]
            return AnomalyEvent(
                frame_number=self._current_frame,
                track_id=track_id,
                anomaly_type="atypical_combination",
                explanation=f"{explanation} (postura inconsistente)",
                activity=activity.get_display_label(),
                posture=posture
            )
        
        return None
    
    def _check_rapid_transitions(
        self,
        track_id: int,
        activity: ActivityResult
    ) -> Optional[AnomalyEvent]:
        """
        Check for rapid activity transitions.
        
        Args:
            track_id: Person ID
            activity: Current activity result
            
        Returns:
            AnomalyEvent if rapid transitions detected
        """
        # Initialize history for this person
        if track_id not in self._activity_history:
            window_size = self._config.summary.ANOMALY_TRANSITION_WINDOW
            self._activity_history[track_id] = deque(maxlen=window_size)
        
        history = self._activity_history[track_id]
        current_label = activity.get_display_label()
        
        # Add current activity
        history.append(current_label)
        
        # Check if we have enough history
        min_frames = self._config.summary.ANOMALY_MIN_STABLE_FRAMES
        if len(history) < min_frames:
            return None
        
        # Count unique activities in window
        unique_activities = len(set(history))
        max_transitions = self._config.summary.ANOMALY_MAX_TRANSITIONS
        
        if unique_activities > max_transitions:
            return AnomalyEvent(
                frame_number=self._current_frame,
                track_id=track_id,
                anomaly_type="rapid_transition",
                explanation=f"Transições rápidas: {unique_activities} atividades em {len(history)} frames",
                activity=current_label,
                posture=activity.posture
            )
        
        return None
    
    def get_summary(self) -> VideoStatistics:
        """
        Get complete video statistics.
        
        Returns:
            VideoStatistics object with all collected data
        """
        return self._statistics
    
    def reset(self) -> None:
        """Reset all statistics (useful for testing)."""
        self._statistics = VideoStatistics()
        self._last_keypoints.clear()
        self._activity_history.clear()
        self._current_frame = 0
