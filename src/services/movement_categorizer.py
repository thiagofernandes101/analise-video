"""
Movement categorizer service.

Analyzes frame history to generate categorized movement segments.
Handles partial body visibility and detects anomalies.

Single Responsibility: Post-process frame history into movement segments.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter

from models.video_statistics import FrameInfo, MovementSegment, PersonStatistics
from config import Config


# COCO keypoint indices
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_LEFT_EAR = 3
KEYPOINT_RIGHT_EAR = 4
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_ELBOW = 7
KEYPOINT_RIGHT_ELBOW = 8
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12
KEYPOINT_LEFT_KNEE = 13
KEYPOINT_RIGHT_KNEE = 14
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16


class MovementCategorizer:
    """
    Analyzes frame history to produce categorized movement segments.
    
    Handles:
    - Partial body visibility (upper body only, hands only, etc.)
    - Movement smoothing to reduce noise
    - Anomaly detection within segments
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the categorizer.
        
        Args:
            config: Configuration object
        """
        self._config = config or Config()
        
        # Configurable parameters
        self._smoothing_window = 5  # Frames for activity smoothing
        self._min_segment_duration = 15  # Minimum frames for a segment
        self._gap_threshold = 30  # Frames gap to break segments
    
    def categorize(self, person_stats: PersonStatistics) -> None:
        """
        Analyze frame history and populate movement_segments.
        
        Args:
            person_stats: PersonStatistics object with frame_history populated
        """
        if not person_stats.frame_history:
            return
        
        # Step 1: Smooth activities
        smoothed_activities = self._smooth_activities(person_stats.frame_history)
        
        # Step 2: Create segments
        raw_segments = self._create_raw_segments(
            person_stats.frame_history, 
            smoothed_activities
        )
        
        # Step 3: Merge similar adjacent segments
        merged_segments = self._merge_adjacent_segments(raw_segments)
        
        # Step 4: Detect anomalies within segments
        final_segments = self._detect_segment_anomalies(
            merged_segments, 
            person_stats.frame_history
        )
        
        person_stats.movement_segments = final_segments
    
    def _smooth_activities(self, history: List[FrameInfo]) -> List[str]:
        """
        Apply smoothing to reduce noise in activity detection.
        
        Args:
            history: List of FrameInfo objects
            
        Returns:
            List of smoothed activity labels
        """
        if not history:
            return []
        
        activities = []
        for f in history:
            if f.is_tracking_error:
                activities.append("Unknown")
            else:
                activities.append(f.activity or "Unknown")
        smoothed = []
        
        for i in range(len(activities)):
            # Get window around current frame
            start = max(0, i - self._smoothing_window // 2)
            end = min(len(activities), i + self._smoothing_window // 2 + 1)
            window = activities[start:end]
            
            # Most common activity in window
            counter = Counter(window)
            most_common = counter.most_common(1)[0][0]
            smoothed.append(most_common)
        
        return smoothed
    
    def _create_raw_segments(
        self, 
        history: List[FrameInfo], 
        smoothed_activities: List[str]
    ) -> List[MovementSegment]:
        """
        Create raw segments from smoothed activity sequence.
        
        Args:
            history: Original frame history
            smoothed_activities: Smoothed activity labels
            
        Returns:
            List of raw MovementSegment objects
        """
        if not history or not smoothed_activities:
            return []
        
        segments = []
        current_activity = smoothed_activities[0]
        segment_start = history[0].frame
        segment_emotions = [history[0].emotion]
        last_frame = history[0].frame
        
        for i in range(1, len(history)):
            frame_info = history[i]
            activity = smoothed_activities[i]
            frame_gap = frame_info.frame - last_frame
            
            # Check if we should start a new segment
            activity_changed = activity != current_activity
            large_gap = frame_gap > self._gap_threshold
            
            if activity_changed or large_gap:
                # Close current segment
                if last_frame - segment_start >= self._min_segment_duration:
                    dominant_emotion = self._get_dominant_emotion(segment_emotions)
                    segments.append(MovementSegment(
                        start_frame=segment_start,
                        end_frame=last_frame,
                        activity=current_activity,
                        dominant_emotion=dominant_emotion
                    ))
                
                # Start new segment
                current_activity = activity
                segment_start = frame_info.frame
                segment_emotions = [frame_info.emotion]
            else:
                segment_emotions.append(frame_info.emotion)
            
            last_frame = frame_info.frame
        
        # Close final segment
        if last_frame - segment_start >= self._min_segment_duration:
            dominant_emotion = self._get_dominant_emotion(segment_emotions)
            segments.append(MovementSegment(
                start_frame=segment_start,
                end_frame=last_frame,
                activity=current_activity,
                dominant_emotion=dominant_emotion
            ))
        
        return segments
    
    def _merge_adjacent_segments(
        self, 
        segments: List[MovementSegment]
    ) -> List[MovementSegment]:
        """
        Merge adjacent segments with the same activity.
        
        Args:
            segments: List of segments
            
        Returns:
            Merged list of segments
        """
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            # Merge if same activity and close in time
            if (seg.activity == last.activity and 
                seg.start_frame - last.end_frame < self._gap_threshold):
                # Extend the last segment
                last.end_frame = seg.end_frame
            else:
                merged.append(seg)
        
        return merged
    
    def _detect_segment_anomalies(
        self, 
        segments: List[MovementSegment],
        history: List[FrameInfo]
    ) -> List[MovementSegment]:
        """
        Detect anomalies within each segment.
        
        Args:
            segments: List of segments
            history: Original frame history
            
        Returns:
            Segments with anomalies populated
        """
        # Build frame lookup
        frame_lookup: Dict[int, FrameInfo] = {f.frame: f for f in history}
        
        for segment in segments:
            anomalies = []
            
            # Get frames in this segment
            segment_frames = [
                frame_lookup[f] for f in range(segment.start_frame, segment.end_frame + 1)
                if f in frame_lookup
            ]
            
            # Check for rapid movement in static posture using pre-calculated velocity
            anomalies.extend(self._check_movement_intensity(segment, segment_frames))
            
            segment.anomalies = anomalies
        
        return segments
    
    def _check_movement_intensity(
        self, 
        segment: MovementSegment, 
        frames: List[FrameInfo]
    ) -> List[str]:
        """
        Check for unexpected movement intensity using velocity data.
        
        Args:
            segment: Current segment
            frames: Frames in the segment
            
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        
        if not frames:
            return anomalies
            
        # Filter out tracking errors for clean analysis
        valid_frames = [f for f in frames if not f.is_tracking_error]
        
        if len(valid_frames) < 2:
            return anomalies
        
        # Calculate average velocity
        velocities = [f.velocity for f in valid_frames]
        avg_velocity = np.mean(velocities)
        
        # Get threshold from config or default
        # Use same threshold as anomaly detector for consistency
        threshold = getattr(self._config.summary, 'ANOMALY_MOVEMENT_THRESHOLD', 50.0)
        
        activity_lower = segment.activity.lower()
        
        # High movement but static activity
        is_static = "sitting" in activity_lower or "standing" in activity_lower or "laying" in activity_lower
        
        if avg_velocity > threshold and is_static:
            anomalies.append(f"High movement ({avg_velocity:.1f}px/fr) during {segment.activity}")
            
        return anomalies
    
    def _get_dominant_emotion(self, emotions: List[str]) -> str:
        """
        Get the most frequent emotion from a list.
        
        Args:
            emotions: List of emotion strings
            
        Returns:
            Most common emotion, or "Unknown" if empty
        """
        if not emotions:
            return "Unknown"
        
        # Filter out unknown/analyzing
        valid = [e for e in emotions if e and e not in ("Unknown", "Analyzing")]
        
        if not valid:
            return "Unknown"
        
        counter = Counter(valid)
        return counter.most_common(1)[0][0]
    
    def get_visible_body_parts(self, keypoints: np.ndarray) -> Dict[str, bool]:
        """
        Determine which body parts are visible in the keypoints.
        
        Useful for handling partial body visibility (e.g., only upper body).
        
        Args:
            keypoints: COCO format keypoints (17, 2)
            
        Returns:
            Dict with body part visibility flags
        """
        def is_visible(idx: int) -> bool:
            if idx >= len(keypoints):
                return False
            return keypoints[idx][0] > 0 and keypoints[idx][1] > 0
        
        return {
            "head": any(is_visible(i) for i in [KEYPOINT_NOSE, KEYPOINT_LEFT_EYE, KEYPOINT_RIGHT_EYE]),
            "shoulders": is_visible(KEYPOINT_LEFT_SHOULDER) or is_visible(KEYPOINT_RIGHT_SHOULDER),
            "arms": any(is_visible(i) for i in [KEYPOINT_LEFT_ELBOW, KEYPOINT_RIGHT_ELBOW, 
                                                  KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST]),
            "hands": is_visible(KEYPOINT_LEFT_WRIST) or is_visible(KEYPOINT_RIGHT_WRIST),
            "torso": is_visible(KEYPOINT_LEFT_HIP) or is_visible(KEYPOINT_RIGHT_HIP),
            "legs": any(is_visible(i) for i in [KEYPOINT_LEFT_KNEE, KEYPOINT_RIGHT_KNEE,
                                                 KEYPOINT_LEFT_ANKLE, KEYPOINT_RIGHT_ANKLE])
        }
    
    def categorize_partial_body(
        self, 
        keypoints: np.ndarray, 
        visibility: Dict[str, bool]
    ) -> Optional[str]:
        """
        Categorize movement when only partial body is visible.
        
        Args:
            keypoints: COCO format keypoints
            visibility: Body part visibility dict
            
        Returns:
            Activity label or None if cannot determine
        """
        # Only hands visible - could be handshake, waving
        if visibility["hands"] and not visibility["torso"] and not visibility["legs"]:
            left_wrist = keypoints[KEYPOINT_LEFT_WRIST] if KEYPOINT_LEFT_WRIST < len(keypoints) else [0, 0]
            right_wrist = keypoints[KEYPOINT_RIGHT_WRIST] if KEYPOINT_RIGHT_WRIST < len(keypoints) else [0, 0]
            
            # Basic heuristic: if hands are close together, might be handshake
            if left_wrist[0] > 0 and right_wrist[0] > 0:
                hand_distance = np.sqrt(
                    (left_wrist[0] - right_wrist[0])**2 + 
                    (left_wrist[1] - right_wrist[1])**2
                )
                if hand_distance < 50:
                    return "Handshake"
            
            return "Hand Movement"
        
        # Upper body only (ballerina case)
        if visibility["head"] and visibility["shoulders"] and visibility["arms"] and not visibility["legs"]:
            return "Upper Body Movement"
        
        return None
