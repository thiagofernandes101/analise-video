"""
Body Part Tracker - Adaptive movement tracking with visibility detection.

Analyzes movement per body part with:
- Automatic visibility detection
- Activity-aware adaptive thresholds
- Dynamic intensity normalization
"""
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from models.movement_profile import MovementProfileManager
from config import Config


class BodyPartTracker:
    """
    Tracks movement per body part with ADAPTIVE thresholds.
    
    KEY INNOVATION: Adapts to partial body visibility!
    - Only analyzes visible body parts
    - Ignores off-screen or occluded parts
    - Uses activity-aware thresholds for visible parts
    """
    
    # YOLO Pose keypoint indices grouped by body part
    BODYPART_KEYPOINTS = {
        "head": [0, 1, 2, 3, 4],           # nose, eyes, ears
        "shoulders": [5, 6],                # left/right shoulder
        "arms": [7, 8],                     # elbows
        "hands": [9, 10],                   # wrists (proxy for hands)
        "hips": [11, 12],                   # left/right hip
        "legs": [13, 14, 15, 16],          # knees, ankles
    }
    
    def __init__(self, config: Config = None):
        """
        Initialize tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.previous_keypoints: Dict[int, np.ndarray] = {}
        self.visibility_threshold = self.config.movement_detection.KEYPOINT_VISIBILITY_THRESHOLD
        self.min_visible_ratio = self.config.movement_detection.MIN_VISIBLE_KEYPOINTS_RATIO
    
    def detect_visible_bodyparts(self, keypoints: np.ndarray) -> Set[str]:
        """
        Detect which body parts are actually visible in the frame.
        
        A body part is "visible" if at least 50% of its keypoints
        have valid coordinates (> 0).
        
        NOTE: Keypoints are in (17, 2) format [x, y] without confidence scores.
        We detect visibility by checking if coordinates are > 0.
        
        Args:
            keypoints: YOLO keypoints (17, 2) with [x, y] coordinates
            
        Returns:
            Set of visible body part names
        """
        visible_parts = set()
        
        for part_name, keypoint_indices in self.BODYPART_KEYPOINTS.items():
            visible_count = 0
            total_count = len(keypoint_indices)
            
            for idx in keypoint_indices:
                if idx < len(keypoints):
                    x, y = keypoints[idx][0], keypoints[idx][1]
                    # Keypoint is "visible" if coordinates are > 0
                    if x > 0 and y > 0:
                        visible_count += 1
            
            # Part is "visible" if >=50% of keypoints are visible
            if visible_count / total_count >= self.min_visible_ratio:
                visible_parts.add(part_name)
        
        return visible_parts
    
    def analyze_bodypart_movement(
        self,
        track_id: int,
        current_keypoints: np.ndarray,
        activity: str
    ) -> Dict[str, Dict]:
        """
        Analyze movement for each VISIBLE body part.
        
        Args:
            track_id: Person tracking ID
            current_keypoints: Current frame keypoints (17, 2) - [x, y] format
            activity: Detected activity
            
        Returns:
            Dictionary of body part movements and anomalies
        """
        if track_id not in self.previous_keypoints:
            self.previous_keypoints[track_id] = current_keypoints
            return {}
        
        prev_kpts = self.previous_keypoints[track_id]
        
        # STEP 1: Detect which body parts are visible
        visible_parts = self.detect_visible_bodyparts(current_keypoints)
        
        if not visible_parts:
            # No visible body parts - skip analysis
            return {}
        
        # STEP 2: Analyze only visible parts
        results = {}
        
        for part_name in visible_parts:  # Only iterate over VISIBLE parts
            keypoint_indices = self.BODYPART_KEYPOINTS[part_name]
            
            # Calculate average movement for this body part
            movements = []
            valid_points = 0
            
            for idx in keypoint_indices:
                if idx < len(current_keypoints) and idx < len(prev_kpts):
                    # Check if keypoints are visible (coordinates > 0)
                    curr_x, curr_y = current_keypoints[idx][0], current_keypoints[idx][1]
                    prev_x, prev_y = prev_kpts[idx][0], prev_kpts[idx][1]
                    
                    if curr_x > 0 and curr_y > 0 and prev_x > 0 and prev_y > 0:
                        # Calculate Euclidean distance
                        movement = np.linalg.norm(
                            current_keypoints[idx] - prev_kpts[idx]
                        )
                        movements.append(movement)
                        valid_points += 1
            
            if valid_points == 0:
                continue
            
            avg_movement = np.mean(movements)
            max_movement = np.max(movements) if movements else 0
            
            # STEP 3: Get adaptive threshold for this body part
            threshold = MovementProfileManager.get_adaptive_threshold(
                activity_label=activity,
                visible_bodyparts=visible_parts,
                bodypart_name=part_name
            )
            
            # STEP 4: Check if movement is anomalous
            is_anomalous = max_movement > threshold
            
            results[part_name] = {
                "avg_movement": avg_movement,
                "max_movement": max_movement,
                "threshold": threshold,
                "is_anomalous": is_anomalous,
                "valid_keypoints": valid_points,
                "is_visible": True,  # We know it's visible
            }
        
        # Update history
        self.previous_keypoints[track_id] = current_keypoints
        
        return results
    
    def calculate_normalized_intensity(
        self,
        bodypart_analysis: Dict[str, Dict],
        activity: str
    ) -> float:
        """
        Calculate movement intensity score (0-1+) based on visible parts.
        
        This is activity-aware but visibility-adaptive!
        
        Args:
            bodypart_analysis: Movement data for visible parts
            activity: Detected activity
            
        Returns:
            Normalized intensity score (0.0 = no movement, 1.0 = exactly as expected)
        """
        profile = MovementProfileManager.get_profile(activity)
        
        if not bodypart_analysis:
            return 0.0
        
        # Calculate "expected movement" for each visible part
        part_scores = []
        
        for part_name, data in bodypart_analysis.items():
            actual_movement = data["avg_movement"]
            
            # What's the expected movement for this part in this activity?
            base_movement = profile.base_movement_per_intensity[profile.expected_intensity]
            bodypart_factor = profile.bodypart_intensity_factors.get(part_name, 1.0)
            expected_movement = base_movement * bodypart_factor
            
            # Calculate ratio: actual / expected
            if expected_movement > 0:
                ratio = actual_movement / expected_movement
                # Clamp to reasonable range (0.0 to 3.0)
                ratio = min(max(ratio, 0.0), 3.0)
                part_scores.append(ratio)
        
        # Average across all visible parts
        if not part_scores:
            return 0.0
        
        overall_intensity = np.mean(part_scores)
        return overall_intensity
    
    def is_movement_appropriate(
        self,
        bodypart_analysis: Dict[str, Dict],
        activity: str
    ) -> Tuple[bool, str, float]:
        """
        Determine if movement intensity is appropriate for activity.
        
        Works with ANY body part combination!
        
        Returns:
            (is_appropriate, explanation, intensity_score)
        """
        intensity = self.calculate_normalized_intensity(bodypart_analysis, activity)
        
        # Intensity ranges for appropriateness
        min_normal = self.config.movement_detection.MIN_NORMAL_INTENSITY
        max_normal = self.config.movement_detection.MAX_NORMAL_INTENSITY
        
        is_appropriate = min_normal <= intensity <= max_normal
        
        if intensity < min_normal:
            explanation = f"Too little movement for {activity} ({intensity:.1f}x expected)"
        elif intensity > max_normal:
            explanation = f"Excessive movement for {activity} ({intensity:.1f}x expected)"
        else:
            explanation = f"Normal movement for {activity} ({intensity:.1f}x expected)"
        
        return is_appropriate, explanation, intensity
    
    def get_anomalous_parts(self, bodypart_analysis: Dict) -> List[str]:
        """Get list of body parts that moved anomalously."""
        return [
            part for part, data in bodypart_analysis.items()
            if data.get("is_anomalous", False)
        ]
    
    def get_movement_summary(
        self,
        bodypart_analysis: Dict,
        activity: str
    ) -> str:
        """
        Generate human-readable movement summary.
        
        Example: "Hands (250px), Arms (180px) - High intensity for Waving"
        """
        if not bodypart_analysis:
            return "No visible body parts"
        
        parts_summary = []
        for part, data in bodypart_analysis.items():
            avg_mov = data["avg_movement"]
            parts_summary.append(f"{part.capitalize()} ({avg_mov:.0f}px)")
        
        visible_parts = ", ".join(parts_summary[:3])  # Top 3
        intensity = self.calculate_normalized_intensity(bodypart_analysis, activity)
        return f"{visible_parts} - {intensity:.1f}x intensity for {activity}"
