"""
Keypoint Smoother Service.

Implements Kalman Filtering to smooth out jittery keypoints from pose estimation.
"""
import numpy as np
from typing import Dict, Tuple, Optional

class KalmanFilter:
    """
    Simple Constant-Velocity Kalman Filter for 2D Point.
    State: [x, y, dx, dy]
    """
    def __init__(self, dt: float = 1.0, process_noise: float = 0.1, measure_noise: float = 1.0):
        # State transition matrix (F)
        # x_new = x + dx*dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (H)
        # We only measure x and y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise covariance (R)
        self.R = np.eye(2) * measure_noise
        
        # Process noise covariance (Q)
        # Model uncertainty
        self.Q = np.eye(4) * process_noise
        
        # Error covariance (P)
        self.P = np.eye(4) * 1.0
        
        # Initial state (x)
        self.x = np.zeros(4)
        
        # Initialization flag
        self.is_initialized = False

    def predict(self) -> np.ndarray:
        """Predict next state."""
        if not self.is_initialized:
            return np.zeros(2)
            
        # x = F * x
        self.x = self.F @ self.x
        
        # P = F * P * F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:2]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update with new measurement.
        
        Args:
            measurement: [x, y]
            
        Returns:
            Filtered [x, y]
        """
        if not self.is_initialized:
            self.x[:2] = measurement
            self.x[2:] = 0  # Zero initial velocity
            self.P = np.eye(4) * 5.0  # High initial uncertainty
            self.is_initialized = True
            return measurement
            
        # y = z - H * x (Innovation)
        z = measurement
        y = z - (self.H @ self.x)
        
        # S = H * P * H.T + R (Innovation covariance)
        S = self.H @ self.P @ self.H.T + self.R
        
        # K = P * H.T * inv(S) (Kalman gain)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((4, 2))
            
        # x = x + K * y
        self.x = self.x + (K @ y)
        
        # P = (I - K * H) * P
        I = np.eye(4)
        self.P = (I - (K @ self.H)) @ self.P
        
        return self.x[:2]
    
    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.is_initialized = False


class KeypointSmoother:
    """
    Manages Kalman filters for all body keypoints of tracked persons.
    """
    def __init__(self, config=None):
        """
        Initialize smoother.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Tracking filters: {track_id: {keypoint_idx: KalmanFilter}}
        self._filters: Dict[int, Dict[int, KalmanFilter]] = {}
        
        # Configuration (can be moved to config object later)
        self.process_noise = 0.01  # Low values = assume smooth motion
        self.measure_noise = 5.0   # High values = trust prediction more (smoother)
        
    def smooth(self, track_id: int, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to keypoints.
        
        Args:
            track_id: Person ID
            keypoints: Raw keypoints (17, 2)
            
        Returns:
            Smoothed keypoints (17, 2)
        """
        if track_id not in self._filters:
            self._filters[track_id] = {}
            
        person_filters = self._filters[track_id]
        smoothed_kpts = np.zeros_like(keypoints)
        
        for idx in range(len(keypoints)):
            x, y = keypoints[idx]
            
            # If point invalid (0,0), return as is (don't smooth zeros)
            # OR predict if we want to fill gaps? 
            # For now, let's just pass through zeros to respect visibility logic
            if x <= 0 or y <= 0:
                # Reset filter for this point if it reappears later to avoid jump
                if idx in person_filters:
                    person_filters[idx].reset()
                smoothed_kpts[idx] = [0, 0]
                continue
                
            # Initialize filter for this keypoint if needed
            if idx not in person_filters:
                person_filters[idx] = KalmanFilter(
                    process_noise=self.process_noise,
                    measure_noise=self.measure_noise
                )
            
            # Update filter
            smoothed_pos = person_filters[idx].update(np.array([x, y]))
            
            # Ensure valid bounds (not negative)
            smoothed_pos = np.maximum(smoothed_pos, 0)
            smoothed_kpts[idx] = smoothed_pos
            
        return smoothed_kpts
        
    def remove_person(self, track_id: int):
        """Clean up filters for lost person."""
        if track_id in self._filters:
            del self._filters[track_id]
