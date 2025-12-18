"""
Video statistics data models.

Contains dataclasses for tracking video analysis results.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING
from collections import Counter
import numpy as np


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly in the video."""
    frame_number: int
    track_id: int
    anomaly_type: str
    explanation: str
    activity: str
    posture: str
    # Enhanced detection fields
    confidence: float = 0.0  # 0.0-1.0 confidence score
    severity: str = "low"  # low, medium, high
    intensity_score: float = 0.0  # Normalized movement intensity
    
    def get_short_description(self) -> str:
        """
        Get a short description for display.
        
        Returns:
            Formatted string like "Frame 145, ID #1: Movimento abrupto"
        """
        type_labels = {
            'abrupt_movement': 'Movimento abrupto',
            'rapid_transition': 'Transições rápidas',
            'atypical_combination': 'Combinação atípica'
        }
        label = type_labels.get(self.anomaly_type, self.anomaly_type)
        return f"Frame {self.frame_number}, ID #{self.track_id}: {label}"


@dataclass
class FrameInfo:
    """
    Stores keypoint and emotion data for a single frame.
    
    Part of the per-person history for movement categorization.
    """
    frame: int
    emotion: str
    keypoints: np.ndarray  # Shape: (17, 2) for COCO format
    activity: Optional[str] = None  # Activity detected in this frame
    # Kinematic data from VelocityTracker
    velocity: float = 0.0
    is_sudden_movement: bool = False
    is_tracking_error: bool = False


@dataclass
class MovementSegment:
    """
    Represents a categorized segment of movement.
    
    Attributes:
        start_frame: Starting frame of the segment
        end_frame: Ending frame of the segment
        activity: Primary activity during this segment (e.g., "Walking")
        dominant_emotion: Most frequent emotion during this segment
        anomalies: List of anomaly descriptions within this segment
    """
    start_frame: int
    end_frame: int
    activity: str
    dominant_emotion: str = "Unknown"
    anomalies: List[str] = field(default_factory=list)
    
    def get_duration_display(self, fps: float = 30.0) -> str:
        """Get human-readable duration display."""
        start_sec = self.start_frame / fps
        end_sec = self.end_frame / fps
        start_str = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}"
        end_str = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
        return f"[{start_str} - {end_str}]"
    
    def has_anomalies(self) -> bool:
        """Check if segment contains anomalies."""
        return len(self.anomalies) > 0


@dataclass
class PersonStatistics:
    """
    Statistics for a single tracked person.
    
    Attributes:
        track_id: Person ID (YOLO tracking ID)
        emotions: List of all unique emotions detected for this person
        activities: List of all unique activities detected for this person
        anomalies: List of anomaly events for this person
        frame_count: Number of frames this person was tracked
    """
    track_id: int
    emotions: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    anomalies: List[AnomalyEvent] = field(default_factory=list)
    frame_count: int = 0
    # New fields for movement categorization
    frame_history: List[FrameInfo] = field(default_factory=list)
    movement_segments: List[MovementSegment] = field(default_factory=list)
    
    def add_emotion(self, emotion: str) -> None:
        """Add emotion if not already recorded."""
        if emotion and emotion not in self.emotions and emotion != "Unknown":
            self.emotions.append(emotion)
    
    def add_activity(self, activity: str) -> None:
        """Add activity if not already recorded."""
        if activity and activity not in self.activities and activity != "Unknown":
            self.activities.append(activity)
    
    def add_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Add an anomaly event."""
        self.anomalies.append(anomaly)
    
    def get_emotions_display(self) -> str:
        """Get comma-separated emotions for display."""
        if not self.emotions:
            return "Nenhuma emoção detectada"
        return ", ".join(self.emotions)
    
    def get_activities_display(self) -> str:
        """Get comma-separated activities for display."""
        if not self.activities:
            return "Nenhuma atividade detectada"
        return ", ".join(self.activities)
    
    def add_frame_info(
        self, 
        frame: int, 
        emotion: str, 
        keypoints: np.ndarray, 
        activity: Optional[str] = None,
        velocity: float = 0.0,
        is_sudden_movement: bool = False,
        is_tracking_error: bool = False
    ) -> None:
        """Add frame data to history."""
        self.frame_history.append(FrameInfo(
            frame=frame,
            emotion=emotion,
            keypoints=keypoints.copy() if keypoints is not None else np.zeros((17, 2)),
            activity=activity,
            velocity=velocity,
            is_sudden_movement=is_sudden_movement,
            is_tracking_error=is_tracking_error
        ))
    
    def get_segments_display(self, fps: float = 30.0) -> List[str]:
        """Get formatted list of movement segments for display."""
        if not self.movement_segments:
            return ["Nenhum segmento de movimento categorizado"]
        
        result = []
        for seg in self.movement_segments:
            line = f"{seg.get_duration_display(fps)} {seg.activity} ({seg.dominant_emotion})"
            if seg.has_anomalies():
                line += f" ⚠️ {', '.join(seg.anomalies)}"
            result.append(line)
        return result


@dataclass
class VideoStatistics:
    """
    Comprehensive video analysis statistics.
    
    Aggregates all statistics from video processing including per-person
    tracking, anomalies, and global activity/emotion distributions.
    
    Attributes:
        total_frames: Total number of frames processed
        person_stats: Per-person statistics keyed by track_id
        all_anomalies: Complete list of all anomalies detected
        activity_distribution: Count of each activity type
        emotion_distribution: Count of each emotion type
    """
    total_frames: int = 0
    person_stats: Dict[int, PersonStatistics] = field(default_factory=dict)
    all_anomalies: List[AnomalyEvent] = field(default_factory=list)
    activity_distribution: Dict[str, int] = field(default_factory=dict)
    emotion_distribution: Dict[str, int] = field(default_factory=dict)
    
    def get_person_count(self) -> int:
        """Get total number of unique persons tracked."""
        return len(self.person_stats)
    
    def get_anomaly_count(self) -> int:
        """Get total number of anomalies detected."""
        return len(self.all_anomalies)
    
    def get_top_activities(self, limit: int = 5) -> List[tuple[str, int]]:
        """
        Get top N activities by frequency.
        
        Args:
            limit: Maximum number of activities to return
            
        Returns:
            List of (activity, count) tuples sorted by count descending
        """
        sorted_activities = sorted(
            self.activity_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_activities[:limit]
    
    def get_top_emotions(self, limit: int = 5) -> List[tuple[str, int]]:
        """
        Get top N emotions by frequency.
        
        Args:
            limit: Maximum number of emotions to return
            
        Returns:
            List of (emotion, count) tuples sorted by count descending
        """
        sorted_emotions = sorted(
            self.emotion_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:limit]
    
    def get_sorted_persons(self) -> List[PersonStatistics]:
        """
        Get list of all persons sorted by track_id.
        
        Returns:
            List of PersonStatistics sorted by track_id
        """
        return sorted(self.person_stats.values(), key=lambda p: p.track_id)
