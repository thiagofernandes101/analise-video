"""
Face renderer for visualizing face detections and emotions.

Single Responsibility: Render face bounding boxes and emotion labels.
"""
import cv2 as cv
import numpy as np
from typing import List, Dict

from models.face_detection import FaceDetection
from models.emotion_result import EmotionResult
from config import Config


class FaceRenderer:
    """Renders face detection boxes and emotion labels on frames."""
    
    def __init__(self, config: Config = Config()):
        """
        Initialize face renderer.
        
        Args:
            config: Configuration object
        """
        self._config = config
    
    def render(
        self,
        frame: np.ndarray,
        faces: List[FaceDetection],
        emotions: Dict[int, EmotionResult]
    ) -> None:
        """
        Render face boxes and emotions on frame.
        
        Args:
            frame: Image frame to draw on (modified in-place)
            faces: List of detected faces
            emotions: Dict mapping person_id to emotion result
        """
        for face in faces:
            self._draw_face_box(frame, face)
            self._draw_emotion_label(frame, face, emotions)
    
    def _draw_face_box(self, frame: np.ndarray, face: FaceDetection) -> None:
        """Draw bounding box around face."""
        x, y, width, height = face.bounding_box.to_xywh_tuple()
        
        cv.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            self._config.visualization.COLOR_FACE_BOX,
            self._config.visualization.FONT_THICKNESS_BOLD
        )
    
    def _draw_emotion_label(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        emotions: Dict[int, EmotionResult]
    ) -> None:
        """Draw emotion label near face."""
        emotion_text = "?"
        
        if face.has_person_assignment and face.person_id in emotions:
            emotion_result = emotions[face.person_id]
            emotion_text = emotion_result.emotion
        
        x, y, width, height = face.bounding_box.to_xywh_tuple()
        label_y = y - 10
        
        # Move label below face if too close to top
        if label_y < 20:
            label_y = y + height + 20
        
        cv.putText(
            frame,
            emotion_text,
            (x, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            self._config.visualization.FONT_SCALE_LARGE,
            self._config.visualization.COLOR_FACE_BOX,
            self._config.visualization.FONT_THICKNESS_BOLD
        )
