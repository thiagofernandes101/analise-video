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
    
    def filter_tracking_errors(
        self,
        prev_keypoints: np.ndarray,
        curr_keypoints: np.ndarray,
        bbox_size: float = None
    ) -> Tuple[bool, List[int]]:
        """
        Filter out impossible keypoint jumps (likely tracking errors).
        
        A keypoint jump is "impossible" if it moves more than:
        - MAX_VALID_VELOCITY_PX_PER_FRAME (absolute)
        - MAX_KEYPOINT_JUMP_RATIO * bbox_size (relative)
        
        Args:
            prev_keypoints: Previous frame keypoints (17, 2)
            curr_keypoints: Current frame keypoints (17, 2)
            bbox_size: Optional bounding box diagonal size for relative check
            
        Returns:
            (is_valid, invalid_keypoint_indices)
        """
        max_velocity = self.config.movement_detection.MAX_VALID_VELOCITY_PX_PER_FRAME
        max_ratio = self.config.movement_detection.MAX_KEYPOINT_JUMP_RATIO
        
        invalid_indices = []
        
        for i in range(min(len(prev_keypoints), len(curr_keypoints))):
            prev_x, prev_y = prev_keypoints[i][0], prev_keypoints[i][1]
            curr_x, curr_y = curr_keypoints[i][0], curr_keypoints[i][1]
            
            # Skip if either keypoint is not visible
            if prev_x <= 0 or prev_y <= 0 or curr_x <= 0 or curr_y <= 0:
                continue
            
            # Calculate movement
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            # Check absolute threshold
            if distance > max_velocity:
                invalid_indices.append(i)
                continue
            
            # Check relative threshold if bbox provided
            if bbox_size and bbox_size > 0:
                if distance > max_ratio * bbox_size:
                    invalid_indices.append(i)
        
        # Track is invalid if more than 30% of visible keypoints jumped
        visible_count = sum(
            1 for i in range(len(curr_keypoints))
            if curr_keypoints[i][0] > 0 and curr_keypoints[i][1] > 0
        )
        
        is_valid = len(invalid_indices) < (visible_count * 0.3) if visible_count > 0 else True
        
        return is_valid, invalid_indices
    
    def calculate_visibility_confidence(self, visible_parts: Set[str]) -> float:
        """
        Calculate confidence score based on body part visibility.
        
        Lower confidence when fewer body parts are visible, as this
        increases uncertainty in anomaly detection.
        
        Confidence formula:
        - 100% visible (6 parts) = 1.0
        - 50% visible (3 parts) = ~0.65
        - 33% visible (2 parts) = ~0.53
        
        Args:
            visible_parts: Set of visible body part names
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        total_parts = 6  # head, shoulders, arms, hands, hips, legs
        visible_count = len(visible_parts)
        
        # Get configuration
        base = self.config.movement_detection.VISIBILITY_CONFIDENCE_BASE
        scale = self.config.movement_detection.VISIBILITY_CONFIDENCE_SCALE
        min_parts = self.config.movement_detection.MIN_VISIBLE_PARTS_FOR_DETECTION
        
        # If fewer than minimum parts visible, return very low confidence
        if visible_count < min_parts:
            return 0.2
        
        # Linear interpolation: base + scale * (visible_ratio)
        visible_ratio = visible_count / total_parts
        confidence = base + scale * visible_ratio
        
        return min(1.0, confidence)
    
    def get_corrected_keypoints(
        self,
        prev_keypoints: np.ndarray,
        curr_keypoints: np.ndarray,
        invalid_indices: List[int]
    ) -> np.ndarray:
        """
        Get corrected keypoints by replacing invalid ones with previous values.
        
        This is useful for smoothing out tracking errors.
        
        Args:
            prev_keypoints: Previous frame keypoints
            curr_keypoints: Current frame keypoints
            invalid_indices: Indices of keypoints to replace
            
        Returns:
            Corrected keypoints array
        """
        corrected = curr_keypoints.copy()
        
        for idx in invalid_indices:
            if idx < len(prev_keypoints) and idx < len(corrected):
                corrected[idx] = prev_keypoints[idx]
        
        return corrected

