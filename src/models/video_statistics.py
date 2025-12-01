"""
Video statistics data models.

These models represent comprehensive video analysis statistics including
per-person tracking, anomalies, and aggregate metrics.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class AnomalyEvent:
    """
    Represents a single anomaly detection event.
    
    Attributes:
        frame_number: Frame where anomaly was detected
        track_id: Person ID (YOLO tracking ID)
        anomaly_type: Type of anomaly (abrupt_movement, rapid_transition, atypical_combination)
        explanation: Human-readable explanation of why this is anomalous
        activity: Activity label during anomaly
        posture: Posture label during anomaly
    """
    frame_number: int
    track_id: int
    anomaly_type: str
    explanation: str
    activity: str
    posture: str
    
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
