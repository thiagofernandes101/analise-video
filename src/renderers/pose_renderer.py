"""
Pose renderer for visualizing person tracking and keypoints.

Single Responsibility: Render person bounding boxes and pose keypoints.
"""
import cv2 as cv
import numpy as np
from typing import List, Dict, Optional

from models.person_tracking import PersonTracking
from models.activity_result import ActivityResult
from config import Config


class PoseRenderer:
    """Renders person tracking boxes and pose keypoints on frames."""
    
    def __init__(self, config: Config = Config()):
        """
        Initialize pose renderer.
        
        Args:
            config: Configuration object
        """
        self._config = config
    
    def render(
        self,
        frame: np.ndarray,
        persons: List[PersonTracking],
        activities: Optional[Dict[int, ActivityResult]] = None
    ) -> None:
        """
        Render person boxes and keypoints on frame.
        
        Args:
            frame: Image frame to draw on (modified in-place)
            persons: List of tracked persons
            activities: Optional dict mapping person_id to activity
        """
        for person in persons:
            self._draw_person_box(frame, person)
            self._draw_keypoints(frame, person)
            self._draw_person_label(frame, person, activities)
    
    def _draw_person_box(self, frame: np.ndarray, person: PersonTracking) -> None:
        """Draw bounding box around person."""
        (frame_height, frame_width) = frame.shape[:2]
        
        x1, y1, x2, y2 = person.bounding_box.to_xyxy_tuple()
        
        # Clamp coordinates to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)
        
        cv.rectangle(
            frame, 
            (x1, y1), 
            (x2, y2), 
            self._config.visualization.COLOR_PERSON_BOX,
            self._config.visualization.FONT_THICKNESS_NORMAL
        )
    
    def _draw_keypoints(self, frame: np.ndarray, person: PersonTracking) -> None:
        """Draw pose keypoints."""
        for keypoint in person.keypoints:
            keypoint_x, keypoint_y = keypoint
            
            if keypoint_x > 0 and keypoint_y > 0:
                cv.circle(
                    frame,
                    (int(keypoint_x), int(keypoint_y)),
                    3,
                    self._config.visualization.COLOR_KEYPOINT,
                    -1
                )
    
    def _draw_person_label(
        self,
        frame: np.ndarray,
        person: PersonTracking,
        activities: Optional[Dict[int, ActivityResult]]
    ) -> None:
        """Draw ID and activity label above person box."""
        label = f"ID: {person.track_id}"
        
        if activities and person.track_id in activities:
            activity = activities[person.track_id]
            label += f" | {activity.get_display_label()}"
        
        x1, y1, _, _ = person.bounding_box.to_xyxy_tuple()
        label_y = y1 - 10
        
        # Move label inside box if too close to top
        if label_y < 20:
            label_y = y1 + 20
        
        cv.putText(
            frame,
            label,
            (x1, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            self._config.visualization.FONT_SCALE_MEDIUM,
            self._config.visualization.COLOR_PERSON_BOX,
            self._config.visualization.FONT_THICKNESS_NORMAL
        )
