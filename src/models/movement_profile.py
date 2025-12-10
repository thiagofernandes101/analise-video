"""
Movement Profile Model - Defines activity-based movement expectations.

Provides adaptive thresholds based on:
- Detected activity type
- Visible body parts
- Movement intensity levels
"""
from dataclasses import dataclass
from typing import Dict, Set
from enum import Enum


class MovementIntensity(Enum):
    """Movement intensity levels for activities."""
    MINIMAL = "minimal"      # Nearly stationary (standing, posing)
    LOW = "low"             # Small movements (sitting, typing)
    MODERATE = "moderate"   # Regular movement (walking, talking)
    HIGH = "high"           # Active movement (waving, exercising)
    VERY_HIGH = "very_high" # Vigorous movement (dancing, running)


@dataclass
class ActivityMovementProfile:
    """
    Movement profile for a specific activity.
    
    Uses relative intensity factors instead of absolute pixel thresholds
    to adapt to partial body visibility.
    """
    activity_name: str
    expected_intensity: MovementIntensity
    bodypart_intensity_factors: Dict[str, float]  # Multipliers per body part
    base_movement_per_intensity: Dict[MovementIntensity, float]


class MovementProfileManager:
    """
    Manages adaptive movement profiles for all activities.
    
    Key Features:
    - Detects which body parts are actually visible
    - Calculates thresholds based on visible parts only
    - Uses relative movement intensity instead of absolute pixels
    """
    
    # Base movement expectations per intensity level (pixels/frame)
    BASE_MOVEMENT = {
        MovementIntensity.MINIMAL: 30,
        MovementIntensity.LOW: 80,
        MovementIntensity.MODERATE: 200,
        MovementIntensity.HIGH: 350,
        MovementIntensity.VERY_HIGH: 500,
    }
    
    # Activity profiles with relative intensity factors
    PROFILES: Dict[str, ActivityMovementProfile] = {
        "Dancing": ActivityMovementProfile(
            activity_name="Dancing",
            expected_intensity=MovementIntensity.VERY_HIGH,
            bodypart_intensity_factors={
                "hands": 2.0,     # Hands move 2x base for dancing
                "arms": 1.8,      # Arms move 1.8x base
                "head": 1.3,      # Head moves 1.3x base
                "legs": 2.2,      # Legs move 2.2x base (highest)
                "hips": 1.5,      # Hips move 1.5x base
                "shoulders": 1.2, # Shoulders move 1.2x base
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Walking": ActivityMovementProfile(
            activity_name="Walking",
            expected_intensity=MovementIntensity.MODERATE,
            bodypart_intensity_factors={
                "hands": 1.2,
                "arms": 1.3,
                "head": 0.8,
                "legs": 2.0,      # Legs move most when walking
                "hips": 1.5,
                "shoulders": 0.9,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Sitting": ActivityMovementProfile(
            activity_name="Sitting",
            expected_intensity=MovementIntensity.LOW,
            bodypart_intensity_factors={
                "hands": 1.5,     # Can gesture while sitting
                "arms": 1.3,      # Can move arms
                "head": 1.2,      # Can look around
                "legs": 0.3,      # Legs should barely move
                "hips": 0.2,      # Hips stable
                "shoulders": 0.8,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Standing": ActivityMovementProfile(
            activity_name="Standing",
            expected_intensity=MovementIntensity.MINIMAL,
            bodypart_intensity_factors={
                "hands": 1.0,
                "arms": 0.8,
                "head": 1.0,
                "legs": 0.3,      # Minimal leg movement
                "hips": 0.2,
                "shoulders": 0.5,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Waving": ActivityMovementProfile(
            activity_name="Waving",
            expected_intensity=MovementIntensity.HIGH,
            bodypart_intensity_factors={
                "hands": 2.5,     # Hand waves a lot!
                "arms": 2.0,      # Arm waves too
                "head": 0.8,      # Head stable
                "legs": 0.3,      # Legs stable (usually sitting/standing)
                "hips": 0.2,
                "shoulders": 1.0,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Hands Up": ActivityMovementProfile(
            activity_name="Hands Up",
            expected_intensity=MovementIntensity.LOW,
            bodypart_intensity_factors={
                "hands": 1.5,     # Hands can adjust position
                "arms": 1.3,
                "head": 0.8,
                "legs": 0.3,
                "hips": 0.2,
                "shoulders": 1.0,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Laying Down": ActivityMovementProfile(
            activity_name="Laying Down",
            expected_intensity=MovementIntensity.MINIMAL,
            bodypart_intensity_factors={
                "hands": 0.8,
                "arms": 0.7,
                "head": 0.9,
                "legs": 0.2,
                "hips": 0.1,
                "shoulders": 0.4,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
        "Unknown": ActivityMovementProfile(
            activity_name="Unknown",
            expected_intensity=MovementIntensity.MODERATE,
            bodypart_intensity_factors={
                # Conservative: all parts have moderate expectations
                "hands": 1.5,
                "arms": 1.3,
                "head": 1.2,
                "legs": 1.5,
                "hips": 1.0,
                "shoulders": 1.0,
            },
            base_movement_per_intensity=BASE_MOVEMENT,
        ),
    }
    
    @classmethod
    def get_adaptive_threshold(
        cls,
        activity_label: str,
        visible_bodyparts: Set[str],
        bodypart_name: str
    ) -> float:
        """
        Calculate adaptive movement threshold for a body part.
        
        ADAPTIVE LOGIC:
        1. Get activity profile
        2. Check if body part is visible
        3. Calculate threshold: base × intensity × bodypart_factor × margin
        4. If body part not visible, return very high threshold (ignore it)
        
        Args:
            activity_label: Detected activity (e.g., "Standing, Walking")
            visible_bodyparts: Which body parts are actually in frame
            bodypart_name: Body part to get threshold for
            
        Returns:
            Movement threshold in pixels/frame
        """
        profile = cls.get_profile(activity_label)
        
        # If this body part isn't visible, return very high threshold
        # (we can't judge movement of invisible parts)
        if bodypart_name not in visible_bodyparts:
            return 999999.0  # Effectively infinite
        
        # Calculate adaptive threshold
        base_movement = profile.base_movement_per_intensity[profile.expected_intensity]
        bodypart_factor = profile.bodypart_intensity_factors.get(bodypart_name, 1.0)
        
        threshold = base_movement * bodypart_factor
        
        # Add margin for threshold (1.5x for max acceptable movement)
        return threshold * 1.5
    
    @classmethod
    def get_profile(cls, activity_label: str) -> ActivityMovementProfile:
        """
        Get movement profile for activity, with fallback logic.
        
        Args:
            activity_label: Activity string (may be compound like "Standing, Walking")
            
        Returns:
            ActivityMovementProfile for the activity
        """
        # Extract primary activity from compound labels like "Standing, Walking"
        primary_activity = activity_label.split(",")[0].strip()
        
        # Check for exact match
        if primary_activity in cls.PROFILES:
            return cls.PROFILES[primary_activity]
        
        # Check for partial matches
        for profile_name, profile in cls.PROFILES.items():
            if profile_name.lower() in activity_label.lower():
                return profile
        
        # Default to Unknown
        return cls.PROFILES["Unknown"]
