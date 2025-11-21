"""
Visualizer - refactored to orchestrate specialized renderers.

Single Responsibility: Coordinate rendering and display management.
"""
import cv2 as cv
import os
import traceback
import numpy as np
from typing import List, Dict

from models.person_tracking import PersonTracking
from models.face_detection import FaceDetection
from models.activity_result import ActivityResult
from models.emotion_result import EmotionResult
from renderers.pose_renderer import PoseRenderer
from renderers.face_renderer import FaceRenderer
from renderers.hud_renderer import HudRenderer
from config import Config


class Visualizer:
    """
    Orchestrates visualization by delegating to specialized renderers.
    
    Follows Single Responsibility Principle and Open/Closed Principle.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize visualizer with renderer components.
        
        Args:
            config: Configuration object
        """
        self._pose_renderer = PoseRenderer(config)
        self._face_renderer = FaceRenderer(config)
        self._hud_renderer = HudRenderer(config)
        self._config = config
    
    def render_frame(
        self,
        frame: np.ndarray,
        persons: List[PersonTracking],
        faces: List[FaceDetection],
        activities: Dict[int, ActivityResult],
        emotions: Dict[int, EmotionResult]
    ) -> None:
        """
        Render all visual elements on the frame.
        
        Args:
            frame: Image frame to render on (modified in-place)
            persons: List of tracked persons
            faces: List of detected faces
            activities: Dict mapping person_id to activity
            emotions: Dict mapping person_id to emotion
        """
        self._pose_renderer.render(frame, persons, activities)
        self._face_renderer.render(frame, faces, emotions)
        self._hud_renderer.render(frame, persons, activities, emotions)
    
    def show_frame(self, frame: np.ndarray, frame_number: int) -> bool:
        """
        Display frame in window or log progress.
        
        Args:
            frame: Image frame to display
            frame_number: Current frame number
            
        Returns:
            True if user requested quit (pressed 'q'), False otherwise
        """
        try:
            if os.environ.get('DISPLAY'):
                cv.imshow('Video Analysis', frame)
                return cv.waitKey(1) == ord('q')
            else:
                if frame_number % 30 == 0:
                    print(f"Processing frame {frame_number}...")
                return False
        except Exception:
            print("Error displaying frame (cv.imshow):")
            traceback.print_exc()
            os.environ.pop('DISPLAY', None)
            return False
    
    def close(self) -> None:
        """Close all display windows."""
        cv.destroyAllWindows()
