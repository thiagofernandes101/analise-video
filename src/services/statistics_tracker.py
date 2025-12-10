"""
Statistics tracker service.

Responsible for collecting and aggregating video analysis statistics during
processing. Implements adaptive anomaly detection with multi-layer analysis.

Single Responsibility: Statistics collection and anomaly detection.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import deque

from models.person_tracking import PersonTracking
from models.activity_result import ActivityResult
from models.emotion_result import EmotionResult
from models.video_statistics import VideoStatistics, PersonStatistics, AnomalyEvent
from models.anomaly_score import AnomalyScorer
from services.bodypart_tracker import BodyPartTracker
from services.statistical_detector import StatisticalAnomalyDetector
from services.movement_categorizer import MovementCategorizer
from config import Config


class StatisticsTracker:
    """
    Tracks comprehensive video analysis statistics with adaptive anomaly detection.
    
    Collects data throughout video processing including per-person tracking,
    multi-layer anomaly detection, and global activity/emotion distributions.
    
    Follows Single Responsibility Principle: Only handles statistics tracking.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize statistics tracker with adaptive detection systems.
        
        Args:
            config: Configuration object
        """
        self._config = config or Config()
        self._statistics = VideoStatistics()
        
        # NEW: Adaptive detection systems
        self._bodypart_tracker = BodyPartTracker(self._config)
        self._statistical_detector = StatisticalAnomalyDetector(self._config)
        
        # Legacy tracking data (still used for simple checks)
        self._last_keypoints: Dict[int, np.ndarray] = {}
        self._activity_history: Dict[int, deque] = {}
        
        # Movement categorizer for post-processing
        self._movement_categorizer = MovementCategorizer(self._config)
        
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
            else:
                emotion_label = "Unknown"
            
            # Collect frame history for movement categorization
            activity_label = activities[track_id].get_display_label() if track_id in activities else None
            person_stats.add_frame_info(
                frame=frame_number,
                emotion=emotion_label,
                keypoints=person.keypoints,
                activity=activity_label
            )
    
    def _detect_anomaly(
        self,
        person: PersonTracking,
        activity: ActivityResult
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly using multi-layer adaptive detection.
        
        Layers:
        1. Body part visibility & movement analysis
        2. Statistical outlier detection
        3. Intensity appropriateness check
        4. Multi-signal anomaly scoring
        
        Args:
            person: Person tracking data
            activity: Current activity result
            
        Returns:
            AnomalyEvent if anomaly detected with confidence >= threshold, None otherwise
        """
        track_id = person.track_id
        activity_label = activity.get_display_label()
        
        # Skip if no previous keypoints
        if track_id not in self._last_keypoints:
            self._last_keypoints[track_id] = person.keypoints.copy()
            return None
        
        # LAYER 1: Body part movement analysis
        bodypart_analysis = self._bodypart_tracker.analyze_bodypart_movement(
            track_id=track_id,
            current_keypoints=person.keypoints,
            activity=activity_label
        )
        
        if not bodypart_analysis:
            # No visible body parts - skip
            self._last_keypoints[track_id] = person.keypoints.copy()
            return None
        
        # Calculate total movement (for statistical detection)
        total_movement = self._calculate_movement(
            self._last_keypoints[track_id],
            person.keypoints
        )
        
        # Get anomalous body parts
        anomalous_parts = self._bodypart_tracker.get_anomalous_parts(bodypart_analysis)
        
        # Check intensity appropriateness
        is_appropriate, intensity_explanation, intensity_score = \
            self._bodypart_tracker.is_movement_appropriate(bodypart_analysis, activity_label)
        
        # LAYER 2: Statistical outlier detection
        is_outlier, z_score, stats = self._statistical_detector.update_and_detect(
            track_id=track_id,
            current_movement=total_movement
        )
        
        # LAYER 3: Simple threshold check (for extreme cases)
        threshold = self._config.summary.ANOMALY_MOVEMENT_THRESHOLD
        activity_exceeded = total_movement > threshold
        
        # LAYER 4: Calculate comprehensive anomaly score
        anomaly_score = AnomalyScorer.calculate_score(
            activity_exceeded=activity_exceeded,
            bodypart_anomalies=anomalous_parts,
            is_statistical_outlier=is_outlier,
            z_score=z_score,
            intensity_appropriate=is_appropriate,
            intensity_score=intensity_score,
            activity=activity_label,
            movement=total_movement
        )
        
        # Store keypoints for next frame
        self._last_keypoints[track_id] = person.keypoints.copy()
        
        # Only return anomaly if confidence >= threshold
        min_confidence = self._config.movement_detection.MIN_ANOMALY_CONFIDENCE
        if not anomaly_score.is_anomaly or anomaly_score.confidence < min_confidence:
            return None
        
        # Create enhanced anomaly event
        return AnomalyEvent(
            frame_number=self._current_frame,
            track_id=track_id,
            anomaly_type=anomaly_score.anomaly_type.value if anomaly_score.anomaly_type else "unknown",
            explanation=anomaly_score.explanation,
            activity=activity_label,
            posture=activity.posture,
            confidence=anomaly_score.confidence,
            severity=anomaly_score.severity,
            intensity_score=intensity_score
        )
    
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
        Get complete video statistics with movement categorization.
        
        Runs post-processing to categorize movements for each person.
        
        Returns:
            VideoStatistics object with all collected data
        """
        # Run movement categorization for all persons
        for person_stats in self._statistics.person_stats.values():
            self._movement_categorizer.categorize(person_stats)
        
        return self._statistics
    
    def reset(self) -> None:
        """Reset all statistics (useful for testing)."""
        self._statistics = VideoStatistics()
        self._last_keypoints.clear()
        self._activity_history.clear()
        self._current_frame = 0
