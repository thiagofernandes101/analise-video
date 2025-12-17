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
from services.velocity_tracker import VelocityTracker, MovementDerivatives
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
        
        # Adaptive detection systems
        # Adaptive detection systems
        self._bodypart_tracker = BodyPartTracker(self._config)
        self._statistical_detector = StatisticalAnomalyDetector(self._config)
        self._velocity_tracker = VelocityTracker(self._config)
        
        # Keypoint smoothing (Kalman Filter)
        from services.keypoint_smoother import KeypointSmoother
        self._keypoint_smoother = KeypointSmoother(self._config)
        
        # Action Recognition (Deep Learning)
        from services.action_classifier import ActionClassifier
        self._action_classifier = ActionClassifier(self._config)
        self._keypoint_buffers: Dict[int, deque] = {} # {track_id: deque of keypoints}
        self._last_predicted_actions: Dict[int, str] = {} # {track_id: last_action}
        
        # Legacy tracking data (still used for simple checks)
        self._last_keypoints: Dict[int, np.ndarray] = {}
        self._activity_history: Dict[int, deque] = {}
        
        # Movement categorizer for post-processing
        self._movement_categorizer = MovementCategorizer(self._config)
        
        # Current frame number
        self._current_frame = 0
        
        # Emotion smoothing buffer: {track_id: deque([emotions])}
        self._emotion_buffers: Dict[int, deque] = {}
        self._emotion_window_size = getattr(self._config.summary, 'EMOTION_SMOOTHING_WINDOW', 30)
    
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
            
            # Apply Kalman Smoothing to keypoints
            smoothed_kpts = self._keypoint_smoother.smooth(track_id, person.keypoints)
            person.keypoints = smoothed_kpts # Update person object with smoothed data
            
            person_stats = self._statistics.person_stats[track_id]
            person_stats.frame_count += 1
            
            # Calculate movement and velocity
            prev_keypoints = self._last_keypoints.get(track_id)
            current_movement = 0.0
            
            if prev_keypoints is not None:
                current_movement = self._calculate_movement(prev_keypoints, person.keypoints)
            
            # Update velocity statistics
            derivatives = self._velocity_tracker.update(track_id, current_movement)
            
            # Update activity tracking
            activity_label = None
            if track_id in activities:
                activity = activities[track_id]
                activity_label = activity.get_display_label()
                
                # Add to person stats
                person_stats.add_activity(activity_label)
                
                # Update global distribution
                if activity_label not in self._statistics.activity_distribution:
                    self._statistics.activity_distribution[activity_label] = 0
                self._statistics.activity_distribution[activity_label] += 1
                
                # Check for anomalies (pass derivatives)
                anomaly = self._detect_anomaly(person, activity, current_movement, derivatives)
                if anomaly:
                    person_stats.add_anomaly(anomaly)
                    self._statistics.all_anomalies.append(anomaly)
            
            # Update emotion tracking
            raw_emotion_label = "Unknown"
            if track_id in emotions:
                raw_emotion_label = emotions[track_id].emotion
            
            # Smooth emotion
            if track_id not in self._emotion_buffers:
                self._emotion_buffers[track_id] = deque(maxlen=self._emotion_window_size)
            
            # Don't buffer 'Analyzing' or 'Unknown' to keep signal clear
            if raw_emotion_label not in ("Unknown", "Analyzing"):
                self._emotion_buffers[track_id].append(raw_emotion_label)
            
            # Determine smoothed emotion
            smoothed_emotion = "Unknown"
            if self._emotion_buffers[track_id]:
                from collections import Counter
                counter = Counter(self._emotion_buffers[track_id])
                smoothed_emotion = counter.most_common(1)[0][0]
            
            # Fallback to raw if smoothing yielded nothing (and raw is valid)
            if smoothed_emotion == "Unknown" and raw_emotion_label not in ("Unknown", "Analyzing"):
                smoothed_emotion = raw_emotion_label
            
            # --- Action Recognition Inference ---
            if track_id not in self._keypoint_buffers:
                self._keypoint_buffers[track_id] = deque(maxlen=self._action_classifier.window_size)
            
            # Add smoothed keypoints to buffer (must be normalized? for now raw)
            self._keypoint_buffers[track_id].append(person.keypoints)
            
            # --- Action Recognition Inference ---
            if track_id not in self._keypoint_buffers:
                self._keypoint_buffers[track_id] = deque(maxlen=self._action_classifier.window_size)
            
            # Add smoothed keypoints to buffer (must be normalized? for now raw)
            self._keypoint_buffers[track_id].append(person.keypoints)
            
            # Trigger inference occasionally
            if len(self._keypoint_buffers[track_id]) == self._action_classifier.window_size:
                if frame_number % self._action_classifier.inference_interval == 0:
                    buffer_array = np.array(self._keypoint_buffers[track_id])
                    # Run ST-GCN inference
                    raw_action = self._action_classifier.predict(buffer_array)
                    
                    # --- Composite Logic (Ensemble) ---
                    # specific heuristic check for 'hand near face'
                    # (simplified: if wrists [9,10] are close to nose [0] or ears [3,4])
                    hand_near_face = False
                    nose = person.keypoints[0]
                    l_wrist = person.keypoints[9]
                    r_wrist = person.keypoints[10]
                    # Simple distance check (e.g. < 50 pixels)
                    if (np.linalg.norm(nose - l_wrist) < 50) or (np.linalg.norm(nose - r_wrist) < 50):
                         hand_near_face = True
                         
                    frame_context = {'hand_near_face': hand_near_face}
                    
                    from services.composite_action_recognizer import CompositeActionRecognizer
                    refined_action = CompositeActionRecognizer.refine_action(raw_action, smoothed_emotion, frame_context)
                    
                    # Store result
                    self._last_predicted_actions[track_id] = refined_action
            
            # Retrieve last known action (Mitigation: Hold result)
            predicted_action = self._last_predicted_actions.get(track_id)
            
            # --- Use Deep Learning action if available, otherwise fall back to heuristic ---
            # This makes ST-GCN the primary activity source when the model is loaded
            final_activity = predicted_action if predicted_action else activity_label
            
            # --- Collect frame history ---
                
            # Add to person stats if valid
            if smoothed_emotion not in ("Unknown", "Analyzing"):
                person_stats.add_emotion(smoothed_emotion)
                
                # Update global distribution
                if smoothed_emotion not in self._statistics.emotion_distribution:
                    self._statistics.emotion_distribution[smoothed_emotion] = 0
                self._statistics.emotion_distribution[smoothed_emotion] += 1
            
            # Add activity to global distribution (using the final activity)
            if final_activity and final_activity not in ("Unknown", "Unknown (No Model)"):
                person_stats.add_activity(final_activity)
                if final_activity not in self._statistics.activity_distribution:
                    self._statistics.activity_distribution[final_activity] = 0
                self._statistics.activity_distribution[final_activity] += 1
            
            # Collect frame history for movement categorization
            # Use smoothed emotion for history to clean up timeline
            frame_emotion = smoothed_emotion if smoothed_emotion != "Unknown" else raw_emotion_label
            
            # Store the FINAL activity (DL or heuristic) as the main activity field
            # This ensures MovementCategorizer and reports use the correct source
            person_stats.add_frame_info(
                frame=frame_number,
                emotion=frame_emotion,
                keypoints=person.keypoints,
                activity=final_activity,  # Use DL prediction if available
                velocity=derivatives.velocity,
                is_sudden_movement=derivatives.is_sudden_movement,
                is_tracking_error=derivatives.is_tracking_error,
                predicted_action=predicted_action  # Keep original for reference
            )
            
            # Update previous keypoints state
            self._last_keypoints[track_id] = person.keypoints.copy()
    
    def _detect_anomaly(
        self,
        person: PersonTracking,
        activity: ActivityResult,
        total_movement: float,
        derivatives: MovementDerivatives
    ) -> Optional[AnomalyEvent]:
        """
        Detect anomaly using multi-layer adaptive detection.
        
        Layers:
        0. Tracking error filter (impossible keypoint jumps)
        1. Body part visibility & movement analysis
        2. Statistical outlier detection
        3. Intensity appropriateness check
        4. Velocity & acceleration analysis
        5. Temporal trend detection
        6. Visibility confidence (occlusion handling)
        7. Multi-signal anomaly scoring
        
        Args:
            person: Person tracking data
            activity: Current activity result
            total_movement: Pre-calculated total movement
            derivatives: Pre-calculated velocity derivatives
            
        Returns:
            AnomalyEvent if anomaly detected with confidence >= threshold, None otherwise
        """
        track_id = person.track_id
        activity_label = activity.get_display_label()
        
        # Get previous keypoints for body part tracking
        # Note: _last_keypoints is managed by the caller (update method)
        if track_id not in self._last_keypoints:
            return None
        
        prev_keypoints = self._last_keypoints[track_id]
        
        # LAYER 0: Tracking error filter
        is_valid_track, invalid_kpts = self._bodypart_tracker.filter_tracking_errors(
            prev_keypoints=prev_keypoints,
            curr_keypoints=person.keypoints
        )
        
        # If tracking error detected, create immediate anomaly event
        if not is_valid_track:
            return AnomalyEvent(
                frame_number=self._current_frame,
                track_id=track_id,
                anomaly_type="tracking_error",
                explanation=f"Tracking error: {len(invalid_kpts)} keypoints jumped impossibly",
                activity=activity_label,
                posture=activity.posture,
                confidence=0.9,
                severity="high",
                intensity_score=0.0
            )
        
        # LAYER 1: Body part movement analysis
        bodypart_analysis = self._bodypart_tracker.analyze_bodypart_movement(
            track_id=track_id,
            current_keypoints=person.keypoints,
            activity=activity_label
        )
        
        if not bodypart_analysis:
            # No visible body parts - skip
            return None
        
        # Get visible body parts for occlusion handling
        visible_parts = self._bodypart_tracker.detect_visible_bodyparts(person.keypoints)
        
        # LAYER 6: Calculate visibility confidence (occlusion handling)
        visibility_confidence = self._bodypart_tracker.calculate_visibility_confidence(visible_parts)
        
        # If visibility is too low, skip anomaly detection to prevent false positives
        min_parts = self._config.movement_detection.MIN_VISIBLE_PARTS_FOR_DETECTION
        if len(visible_parts) < min_parts:
            return None
        
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
        
        # LAYER 4: Velocity & acceleration analysis
        # (Derivatives passed in)
        is_velocity_anomaly = derivatives.is_sudden_movement or derivatives.is_tracking_error
        velocity_score = derivatives.get_anomaly_score()
        
        # LAYER 5: Temporal trend detection
        is_trend_anomaly, trend_type, trend_strength = self._statistical_detector.detect_trend_anomaly(
            track_id=track_id
        )
        
        # LAYER 7: Calculate comprehensive anomaly score with all signals
        anomaly_score = AnomalyScorer.calculate_score(
            activity_exceeded=activity_exceeded,
            bodypart_anomalies=anomalous_parts,
            is_statistical_outlier=is_outlier,
            z_score=z_score,
            intensity_appropriate=is_appropriate,
            intensity_score=intensity_score,
            activity=activity_label,
            movement=total_movement,
            # New signals
            is_velocity_anomaly=is_velocity_anomaly,
            velocity_score=velocity_score,
            is_trend_anomaly=is_trend_anomaly,
            trend_type=trend_type,
            trend_strength=trend_strength,
            visibility_confidence=visibility_confidence
        )
        # remove self._last_keypoints updates as strictly managed by caller now
        
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
        # Filter out short tracks to reduce noise
        min_frames = getattr(self._config.movement_detection, 'MIN_TRACK_FRAMES', 30)
        
        valid_persons = {}
        for track_id, stats in self._statistics.person_stats.items():
            if stats.frame_count >= min_frames:
                valid_persons[track_id] = stats
        
        self._statistics.person_stats = valid_persons
        
        # Run movement categorization for all valid persons
        for person_stats in self._statistics.person_stats.values():
            self._movement_categorizer.categorize(person_stats)
        
        return self._statistics
    
    def reset(self) -> None:
        """Reset all statistics (useful for testing)."""
        self._statistics = VideoStatistics()
        self._last_keypoints.clear()
        self._activity_history.clear()
        self._velocity_tracker.reset_all()
        self._current_frame = 0

