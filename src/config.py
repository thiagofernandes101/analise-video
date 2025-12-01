"""
Refactored configuration module following Clean Code principles.

Configuration is split into domain-specific classes for better organization.
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VideoConfig:
    """Configuration for video input."""
    
    VIDEO_FILENAME: str = 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    
    @property
    def possible_paths(self) -> List[str]:
        """Get list of possible video file paths."""
        return [
            os.path.join(self.BASE_DIR, '../videos', self.VIDEO_FILENAME),
            os.path.join('/app/videos', self.VIDEO_FILENAME)
        ]
    
    def find_video_path(self) -> Optional[str]:
        """
        Find the first existing video path.
        
        Returns:
            Path to video file if found, None otherwise
        """
        for path in self.possible_paths:
            if os.path.exists(path):
                return path
        return None


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    YOLO_MODEL_NAME: str = "yolov8m-pose.pt"
    FACE_DETECTION_BACKEND: str = 'yolov8m'


@dataclass
class EmotionAnalysisConfig:
    """Configuration for emotion analysis."""
    
    # How many frames to wait before analyzing emotion again for same person
    ANALYSIS_INTERVAL_FRAMES: int = 15
    
    # Minimum confidence threshold to accept emotion (0-100)
    CONFIDENCE_THRESHOLD: float = 40.0
    
    # Maximum queue size for async processing
    QUEUE_MAX_SIZE: int = 2


@dataclass
class ActivityRecognitionConfig:
    """Configuration for activity recognition."""
    
    # Number of frames to keep in history for temporal analysis
    HISTORY_LENGTH_FRAMES: int = 30
    
    # Thresholds for posture detection
    LAYING_DOWN_ANGLE_THRESHOLD: float = 45.0  # degrees
    SITTING_THIGH_RATIO_THRESHOLD: float = 0.45
    
    # Thresholds for action detection
    JUMPING_HEIGHT_RATIO: float = 0.2
    JUMPING_AIR_RATIO: float = 0.1
    WALKING_ANKLE_VARIANCE: float = 50.0
    WAVING_HAND_VARIANCE: float = 100.0
    WAVING_HEAD_VARIANCE: float = 50.0
    HEAD_MOVEMENT_BODY_STABILITY: float = 20.0


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    # Colors (BGR format for OpenCV)
    COLOR_PERSON_BOX: tuple = (255, 255, 0)  # Cyan
    COLOR_FACE_BOX: tuple = (0, 255, 0)  # Green
    COLOR_KEYPOINT: tuple = (0, 0, 255)  # Red
    COLOR_TEXT_WHITE: tuple = (255, 255, 255)
    COLOR_TEXT_GRAY: tuple = (200, 200, 200)
    COLOR_HUD_BACKGROUND: tuple = (0, 0, 0)
    
    # Text settings
    FONT_SCALE_LARGE: float = 0.7
    FONT_SCALE_MEDIUM: float = 0.5
    FONT_THICKNESS_BOLD: int = 2
    FONT_THICKNESS_NORMAL: int = 1
    
    # HUD settings
    HUD_PANEL_WIDTH: int = 300
    HUD_ITEM_HEIGHT: int = 30
    HUD_PADDING: int = 10
    HUD_ALPHA: float = 0.6  # Transparency


@dataclass
class SummaryConfig:
    """Configuration for video analysis summary and anomaly detection."""
    
    # Anomaly detection thresholds
    ANOMALY_MOVEMENT_THRESHOLD: float = 200.0  # pixels per frame (increased for less sensitivity)
    ANOMALY_MIN_STABLE_FRAMES: int = 8  # Minimum frames to maintain activity (increased)
    ANOMALY_MAX_TRANSITIONS: int = 6  # Max activity changes in window (increased)
    ANOMALY_TRANSITION_WINDOW: int = 20  # Frame window for transition check (increased)
    
    # UI settings - Overview
    SUMMARY_WINDOW_WIDTH: int = 700
    SUMMARY_WINDOW_HEIGHT: int = 800
    SUMMARY_FONT_SCALE: float = 0.6
    SUMMARY_FONT_SCALE_LARGE: float = 0.8
    SUMMARY_BAR_MAX_WIDTH: int = 400
    SUMMARY_BUTTON_WIDTH: int = 200
    SUMMARY_BUTTON_HEIGHT: int = 40
    
    # UI settings - Detailed View
    DETAIL_WINDOW_WIDTH: int = 800
    DETAIL_WINDOW_HEIGHT: int = 900
    DETAIL_SCROLL_STEP: int = 30
    DETAIL_INDENT: int = 20
    DETAIL_FONT_SCALE: float = 0.55


class Config:
    """
    Main configuration class providing access to all config domains.
    
    This is the main entry point for configuration, maintaining backward
    compatibility while improving organization.
    """
    
    video = VideoConfig()
    model = ModelConfig()
    emotion = EmotionAnalysisConfig()
    activity = ActivityRecognitionConfig()
    visualization = VisualizationConfig()
    summary = SummaryConfig()
    
    # Backward compatibility properties
    @classmethod
    def get_video_path(cls) -> Optional[str]:
        """Find and return video path (backward compatibility)."""
        return cls.video.find_video_path()
    
    @property
    def YOLO_MODEL_SIZE(self) -> str:
        """Backward compatibility for YOLO model name."""
        return self.model.YOLO_MODEL_NAME
    
    @property
    def EMOTION_ANALYSIS_INTERVAL(self) -> int:
        """Backward compatibility for emotion interval."""
        return self.emotion.ANALYSIS_INTERVAL_FRAMES
    
    @property
    def EMOTION_CONFIDENCE_THRESHOLD(self) -> float:
        """Backward compatibility for emotion threshold."""
        return self.emotion.CONFIDENCE_THRESHOLD
