"""
Statistical Anomaly Detector - Detects movement outliers using statistical methods.

Uses Z-score and MAD (Median Absolute Deviation) to identify movements
that are unusual for a specific person's normal behavior.
"""
import numpy as np
from collections import deque
from typing import Dict, Tuple

from config import Config


class StatisticalAnomalyDetector:
    """
    Detects anomalies using statistical methods (Z-score, MAD).
    
    Tracks movement history per person to identify when current
    movement significantly deviates from their normal pattern.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.window_size = self.config.movement_detection.MOVEMENT_HISTORY_WINDOW
        self.z_threshold = self.config.movement_detection.Z_SCORE_THRESHOLD
        self.modified_z_threshold = self.config.movement_detection.MODIFIED_Z_SCORE_THRESHOLD
        
        # Movement history per person
        self.movement_history: Dict[int, deque] = {}
    
    def update_and_detect(
        self,
        track_id: int,
        current_movement: float
    ) -> Tuple[bool, float, Dict]:
        """
        Update history and detect if current movement is a statistical outlier.
        
        Args:
            track_id: Person ID
            current_movement: Current frame movement in pixels
            
        Returns:
            (is_outlier, z_score, statistics_dict)
        """
        # Initialize history for this person
        if track_id not in self.movement_history:
            self.movement_history[track_id] = deque(maxlen=self.window_size)
        
        history = self.movement_history[track_id]
        
        # Need at least 5 samples for meaningful statistics
        if len(history) < 5:
            history.append(current_movement)
            return False, 0.0, {}
        
        # Calculate statistics
        movements = list(history)
        mean = np.mean(movements)
        std = np.std(movements)
        median = np.median(movements)
        
        # Calculate Z-score
        z_score = 0.0
        if std > 0:
            z_score = (current_movement - mean) / std
        
        # Calculate MAD (Median Absolute Deviation) - more robust to outliers
        mad = np.median(np.abs(movements - median))
        modified_z_score = 0.0
        if mad > 0:
            modified_z_score = 0.6745 * (current_movement - median) / mad
        
        # Update history
        history.append(current_movement)
        
        # Detection criteria (using both Z-score and modified Z-score)
        is_outlier = (
            abs(z_score) > self.z_threshold or 
            abs(modified_z_score) > self.modified_z_threshold
        )
        
        stats = {
            "mean": mean,
            "std": std,
            "median": median,
            "z_score": z_score,
            "modified_z_score": modified_z_score,
            "history_size": len(history),
        }
        
        return is_outlier, z_score, stats
    
    def reset_person(self, track_id: int):
        """Reset movement history for a person."""
        if track_id in self.movement_history:
            del self.movement_history[track_id]
    
    def get_person_statistics(self, track_id: int) -> Dict:
        """
        Get statistical summary for a person.
        
        Args:
            track_id: Person ID
            
        Returns:
            Dictionary with mean, std, median, etc.
        """
        if track_id not in self.movement_history:
            return {}
        
        history = list(self.movement_history[track_id])
        if len(history) < 2:
            return {}
        
        return {
            "mean_movement": np.mean(history),
            "std_movement": np.std(history),
            "median_movement": np.median(history),
            "min_movement": np.min(history),
            "max_movement": np.max(history),
            "sample_count": len(history),
        }
    
    def detect_trend_anomaly(
        self,
        track_id: int,
        window: int = 10
    ) -> Tuple[bool, str, float]:
        """
        Detect if recent movement trend is anomalous.
        
        Analyzes movement patterns to identify:
        - Sudden acceleration (starting to run, panic)
        - Sudden deceleration (stopping abruptly, falling)
        - Erratic movement (unstable tracking or distress)
        
        Args:
            track_id: Person tracking ID
            window: Number of recent frames to analyze
            
        Returns:
            (is_anomaly, trend_type, trend_strength)
        """
        if track_id not in self.movement_history:
            return False, "unknown", 0.0
        
        history = list(self.movement_history[track_id])
        if len(history) < window:
            return False, "insufficient_data", 0.0
        
        recent = history[-window:]
        
        # Calculate velocity differences (acceleration)
        diffs = np.diff(recent)
        mean_acceleration = np.mean(diffs)
        std_acceleration = np.std(diffs)
        
        # Calculate jerk (change in acceleration)
        if len(diffs) >= 2:
            jerks = np.diff(diffs)
            mean_jerk = np.mean(np.abs(jerks))
        else:
            mean_jerk = 0.0
        
        # Classify trend
        trend_type = "stable"
        trend_strength = 0.0
        is_anomaly = False
        
        # High consistent acceleration = suddenly speeding up
        if mean_acceleration > 15:
            trend_type = "accelerating"
            trend_strength = mean_acceleration
            if mean_acceleration > 30:
                is_anomaly = True
        
        # High consistent deceleration = suddenly slowing down
        elif mean_acceleration < -15:
            trend_type = "decelerating"
            trend_strength = abs(mean_acceleration)
            if mean_acceleration < -30:
                is_anomaly = True
        
        # High variance in acceleration = erratic movement
        elif std_acceleration > 20:
            trend_type = "erratic"
            trend_strength = std_acceleration
            is_anomaly = True
        
        # High jerk = very sudden changes
        elif mean_jerk > 25:
            trend_type = "jerky"
            trend_strength = mean_jerk
            is_anomaly = True
        
        return is_anomaly, trend_type, trend_strength
    
    def get_movement_trend(self, track_id: int) -> Tuple[str, float]:
        """
        Get simple movement trend classification.
        
        Args:
            track_id: Person tracking ID
            
        Returns:
            (trend_label, trend_strength)
            trend_label: 'accelerating', 'decelerating', 'stable', 'erratic'
        """
        _, trend_type, trend_strength = self.detect_trend_anomaly(track_id)
        return trend_type, trend_strength

