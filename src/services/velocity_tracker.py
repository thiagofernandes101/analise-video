"""
Velocity Tracker Service - Tracks velocity, acceleration, and jerk for anomaly detection.

Provides derivative-based movement analysis to distinguish:
- Gradual movements (normal acceleration)
- Sudden jerky movements (likely anomalies)
- Tracking errors (physically impossible velocities)
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from collections import deque

from config import Config


@dataclass
class MovementDerivatives:
    """
    Movement derivatives for a single frame.
    
    Attributes:
        velocity: Current velocity in pixels/frame
        acceleration: Change in velocity (pixels/frame²)
        jerk: Change in acceleration (pixels/frame³)
        is_sudden_movement: True if jerk exceeds threshold
        is_tracking_error: True if velocity is physically impossible
    """
    velocity: float
    acceleration: float
    jerk: float
    is_sudden_movement: bool
    is_tracking_error: bool
    
    def get_anomaly_score(self) -> float:
        """Calculate anomaly contribution from derivatives."""
        if self.is_tracking_error:
            return 1.0
        if self.is_sudden_movement:
            return 0.7
        return 0.0


class VelocityTracker:
    """
    Tracks velocity history and calculates derivatives per person.
    
    Key features:
    - Per-person velocity history
    - Acceleration and jerk calculation
    - Sudden movement detection
    - Tracking error identification
    """
    
    def __init__(self, config: Config = None):
        """Initialize velocity tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Velocity history per person (stores recent velocities)
        self.velocity_history: Dict[int, deque] = {}
        
        # Configuration
        self.window_size = getattr(
            self.config.movement_detection, 
            'VELOCITY_HISTORY_WINDOW', 
            10
        )
        self.max_velocity = getattr(
            self.config.movement_detection,
            'MAX_VALID_VELOCITY_PX_PER_FRAME',
            100.0
        )
        self.acceleration_threshold = getattr(
            self.config.movement_detection,
            'ACCELERATION_ANOMALY_THRESHOLD',
            50.0
        )
        self.jerk_threshold = getattr(
            self.config.movement_detection,
            'JERK_ANOMALY_THRESHOLD',
            30.0
        )
    
    def update(
        self,
        track_id: int,
        current_movement: float
    ) -> MovementDerivatives:
        """
        Update velocity history and calculate derivatives.
        
        Args:
            track_id: Person tracking ID
            current_movement: Movement in pixels for this frame
            
        Returns:
            MovementDerivatives with velocity, acceleration, jerk
        """
        # Initialize history for this person
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = deque(maxlen=self.window_size)
        
        history = self.velocity_history[track_id]
        velocity = current_movement  # velocity = distance / 1 frame
        
        # Calculate derivatives
        acceleration = 0.0
        jerk = 0.0
        
        if len(history) >= 1:
            # Acceleration = change in velocity
            prev_velocity = history[-1]
            acceleration = velocity - prev_velocity
            
            if len(history) >= 2:
                # Jerk = change in acceleration
                prev_prev_velocity = history[-2]
                prev_acceleration = prev_velocity - prev_prev_velocity
                jerk = acceleration - prev_acceleration
        
        # Update history
        history.append(velocity)
        
        # Detect anomalies
        is_tracking_error = velocity > self.max_velocity
        is_sudden_movement = (
            abs(jerk) > self.jerk_threshold or 
            abs(acceleration) > self.acceleration_threshold
        )
        
        return MovementDerivatives(
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            is_sudden_movement=is_sudden_movement,
            is_tracking_error=is_tracking_error
        )
    
    def get_velocity_stats(self, track_id: int) -> Dict:
        """
        Get velocity statistics for a person.
        
        Args:
            track_id: Person tracking ID
            
        Returns:
            Dictionary with mean, std, max velocity
        """
        if track_id not in self.velocity_history:
            return {}
        
        history = list(self.velocity_history[track_id])
        if len(history) < 2:
            return {}
        
        return {
            "mean_velocity": np.mean(history),
            "std_velocity": np.std(history),
            "max_velocity": np.max(history),
            "min_velocity": np.min(history),
            "sample_count": len(history),
        }
    
    def get_trend(self, track_id: int) -> Tuple[str, float]:
        """
        Get movement trend classification.
        
        Args:
            track_id: Person tracking ID
            
        Returns:
            (trend_label, trend_strength)
            trend_label: 'accelerating', 'decelerating', 'stable', 'erratic'
        """
        if track_id not in self.velocity_history:
            return "unknown", 0.0
        
        history = list(self.velocity_history[track_id])
        if len(history) < 3:
            return "unknown", 0.0
        
        # Calculate velocity differences
        diffs = np.diff(history)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Classify trend
        if std_diff > abs(mean_diff) * 2:
            return "erratic", std_diff
        elif mean_diff > 5:
            return "accelerating", mean_diff
        elif mean_diff < -5:
            return "decelerating", abs(mean_diff)
        else:
            return "stable", abs(mean_diff)
    
    def reset_person(self, track_id: int) -> None:
        """Reset velocity history for a person."""
        if track_id in self.velocity_history:
            del self.velocity_history[track_id]
    
    def reset_all(self) -> None:
        """Reset all velocity history."""
        self.velocity_history.clear()
