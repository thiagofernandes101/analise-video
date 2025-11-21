"""
HUD renderer for overlay information panel.

Single Responsibility: Render heads-up display with tracking information.
"""
import cv2 as cv
import numpy as np
from typing import List, Dict

from models.person_tracking import PersonTracking
from models.activity_result import ActivityResult
from models.emotion_result import EmotionResult
from config import Config


class HudRenderer:
    """Renders heads-up display panel with tracking information."""
    
    def __init__(self, config: Config = Config()):
        """
        Initialize HUD renderer.
        
        Args:
            config: Configuration object
        """
        self._config = config
    
    def render(
        self,
        frame: np.ndarray,
        persons: List[PersonTracking],
        activities: Dict[int, ActivityResult],
        emotions: Dict[int, EmotionResult]
    ) -> None:
        """
        Render HUD panel on frame.
        
        Args:
            frame: Image frame to draw on (modified in-place)
            persons: List of tracked persons
            activities: Dict mapping person_id to activity
            emotions: Dict mapping person_id to emotion
        """
        if not persons:
            return
        
        panel_dimensions = self._calculate_panel_dimensions(len(persons))
        panel_position = self._calculate_panel_position(frame, panel_dimensions)
        
        self._draw_panel_background(frame, panel_position, panel_dimensions)
        self._draw_panel_header(frame, panel_position)
        self._draw_tracking_items(frame, persons, activities, emotions, panel_position)
    
    def _calculate_panel_dimensions(self, num_items: int) -> tuple[int, int]:
        """Calculate panel width and height based on number of items."""
        panel_width = self._config.visualization.HUD_PANEL_WIDTH
        item_height = self._config.visualization.HUD_ITEM_HEIGHT
        padding = self._config.visualization.HUD_PADDING
        
        panel_height = num_items * item_height + 40  # 40 for header
        
        return (panel_width, panel_height)
    
    def _calculate_panel_position(
        self,
        frame: np.ndarray,
        panel_dimensions: tuple[int, int]
    ) -> tuple[int, int]:
        """Calculate top-right position for panel."""
        (frame_height, frame_width) = frame.shape[:2]
        panel_width, panel_height = panel_dimensions
        padding = self._config.visualization.HUD_PADDING
        
        panel_x = frame_width - panel_width - padding
        panel_y = padding
        
        return (panel_x, panel_y)
    
    def _draw_panel_background(
        self,
        frame: np.ndarray,
        position: tuple[int, int],
        dimensions: tuple[int, int]
    ) -> None:
        """Draw semi-transparent background panel."""
        panel_x, panel_y = position
        panel_width, panel_height = dimensions
        
        overlay = frame.copy()
        cv.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self._config.visualization.COLOR_HUD_BACKGROUND,
            -1
        )
        
        alpha = self._config.visualization.HUD_ALPHA
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _draw_panel_header(self, frame: np.ndarray, position: tuple[int, int]) -> None:
        """Draw panel header text."""
        panel_x, panel_y = position
        padding = self._config.visualization.HUD_PADDING
        
        cv.putText(
            frame,
            "Active Tracks",
            (panel_x + padding, panel_y + 25),
            cv.FONT_HERSHEY_SIMPLEX,
            self._config.visualization.FONT_SCALE_LARGE,
            self._config.visualization.COLOR_TEXT_WHITE,
            self._config.visualization.FONT_THICKNESS_BOLD
        )
    
    def _draw_tracking_items(
        self,
        frame: np.ndarray,
        persons: List[PersonTracking],
        activities: Dict[int, ActivityResult],
        emotions: Dict[int, EmotionResult],
        position: tuple[int, int]
    ) -> None:
        """Draw individual tracking items."""
        panel_x, panel_y = position
        padding = self._config.visualization.HUD_PADDING
        item_height = self._config.visualization.HUD_ITEM_HEIGHT
        
        for index, person in enumerate(persons):
            y_position = panel_y + 60 + (index * item_height)
            
            activity_label = "Unknown"
            if person.track_id in activities:
                activity_label = activities[person.track_id].get_display_label()
            
            emotion_label = "N/A"
            if person.track_id in emotions:
                emotion_label = emotions[person.track_id].emotion
            
            item_text = f"ID {person.track_id}: {activity_label} | {emotion_label}"
            
            cv.putText(
                frame,
                item_text,
                (panel_x + padding, y_position),
                cv.FONT_HERSHEY_SIMPLEX,
                self._config.visualization.FONT_SCALE_MEDIUM,
                self._config.visualization.COLOR_TEXT_GRAY,
                self._config.visualization.FONT_THICKNESS_NORMAL
            )
