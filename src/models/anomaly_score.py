"""
Anomaly Scoring Model - Classifies and scores movement anomalies.

Combines signals from multiple detection layers to provide:
- Confidence scores (0.0-1.0)
- Anomaly type classification
- Severity levels
- Explanations
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    TRACKING_ERROR = "tracking_error"          # Likely YOLO tracking glitch
    IMPOSSIBLE_MOVEMENT = "impossible_movement"  # Physically impossible
    UNEXPECTED_BODYPART = "unexpected_bodypart"  # Wrong part moving
    STATISTICAL_OUTLIER = "statistical_outlier"  # Unusual for this person
    ACTIVITY_MISMATCH = "activity_mismatch"      # Movement doesn't match activity


@dataclass
class AnomalyScore:
    """
    Comprehensive anomaly score with confidence and classification.
    
    Attributes:
        is_anomaly: Whether this is classified as an anomaly
        confidence: Confidence score (0.0-1.0)
        anomaly_type: Classification of anomaly type
        contributing_factors: Which detectors flagged it
        explanation: Human-readable explanation
        severity: "low", "medium", or "high"
    """
    is_anomaly: bool
    confidence: float  # 0.0 to 1.0
    anomaly_type: Optional[AnomalyType]
    contributing_factors: Dict[str, float]  # Which detectors flagged it
    explanation: str
    severity: str  # "low", "medium", "high"
    
    def __post_init__(self):
        """Calculate severity based on confidence."""
        if self.confidence >= 0.8:
            self.severity = "high"
        elif self.confidence >= 0.5:
            self.severity = "medium"
        else:
            self.severity = "low"


class AnomalyScorer:
    """Combines signals from multiple detectors into a single score."""
    
    # Weights for different detection signals (redistributed for new signals)
    WEIGHTS = {
        "activity_threshold": 0.15,
        "bodypart_analysis": 0.20,
        "statistical_outlier": 0.20,
        "intensity_check": 0.15,
        "velocity_anomaly": 0.15,
        "temporal_trend": 0.15,
    }
    
    @classmethod
    def calculate_score(
        cls,
        activity_exceeded: bool,
        bodypart_anomalies: List[str],
        is_statistical_outlier: bool,
        z_score: float,
        intensity_appropriate: bool,
        intensity_score: float,
        activity: str,
        movement: float,
        # New parameters for enhanced detection
        is_velocity_anomaly: bool = False,
        velocity_score: float = 0.0,
        is_trend_anomaly: bool = False,
        trend_type: str = "stable",
        trend_strength: float = 0.0,
        visibility_confidence: float = 1.0,
    ) -> AnomalyScore:
        """
        Calculate comprehensive anomaly score.
        
        Args:
            activity_exceeded: Did movement exceed activity threshold?
            bodypart_anomalies: List of body parts with anomalous movement
            is_statistical_outlier: Is this a statistical outlier?
            z_score: Statistical Z-score
            intensity_appropriate: Is intensity appropriate for activity?
            intensity_score: Normalized intensity (0-3.0+)
            activity: Detected activity
            movement: Total movement in pixels
            
        Returns:
            AnomalyScore with confidence and classification
        """
        signals = {}
        factors = {}
        
        # Signal 1: Activity threshold
        if activity_exceeded:
            signals["activity_threshold"] = 1.0
            factors["Activity threshold exceeded"] = f"{movement:.0f}px"
        else:
            signals["activity_threshold"] = 0.0
        
        # Signal 2: Body part analysis
        bodypart_score = min(len(bodypart_anomalies) / 3.0, 1.0)  # Cap at 3 parts
        signals["bodypart_analysis"] = bodypart_score
        if bodypart_anomalies:
            factors["Anomalous body parts"] = ", ".join(bodypart_anomalies)
        
        # Signal 3: Statistical outlier
        signals["statistical_outlier"] = 1.0 if is_statistical_outlier else 0.0
        if is_statistical_outlier:
            factors["Statistical outlier"] = f"Z-score: {z_score:.2f}"
        
        # Signal 4: Intensity check
        if not intensity_appropriate:
            # How far outside normal range?
            if intensity_score < 0.4:
                signals["intensity_check"] = min((0.4 - intensity_score) / 0.4, 1.0)
            elif intensity_score > 1.6:
                signals["intensity_check"] = min((intensity_score - 1.6) / 1.6, 1.0)
            else:
                signals["intensity_check"] = 0.0
            
            if signals["intensity_check"] > 0.5:
                factors["Intensity mismatch"] = f"{intensity_score:.1f}x (expected 0.4-1.6x)"
        else:
            signals["intensity_check"] = 0.0
        
        # Signal 5: Velocity anomaly (sudden movements, tracking errors)
        if is_velocity_anomaly:
            signals["velocity_anomaly"] = min(velocity_score, 1.0)
            if velocity_score >= 0.7:
                factors["Velocity anomaly"] = f"Score: {velocity_score:.2f}"
        else:
            signals["velocity_anomaly"] = 0.0
        
        # Signal 6: Temporal trend (acceleration patterns)
        if is_trend_anomaly:
            # Map trend strength to 0-1 score
            trend_score = min(trend_strength / 50.0, 1.0)
            signals["temporal_trend"] = trend_score
            factors["Movement trend"] = f"{trend_type} ({trend_strength:.1f})"
        else:
            signals["temporal_trend"] = 0.0
        
        # Calculate weighted confidence score
        confidence = sum(
            signals[key] * cls.WEIGHTS[key]
            for key in cls.WEIGHTS.keys()
        )
        
        # Apply occlusion penalty: reduce confidence when visibility is low
        # This prevents false positives when only part of body is visible
        confidence = cls._apply_occlusion_penalty(confidence, visibility_confidence)
        
        # Determine if it's an anomaly (threshold: 0.4)
        is_anomaly = confidence >= 0.4
        
        # Classify anomaly type
        anomaly_type = cls._classify_anomaly(
            signals, bodypart_anomalies, z_score, movement, intensity_score,
            is_velocity_anomaly, is_trend_anomaly, trend_type
        )
        
        # Generate explanation
        explanation = cls._generate_explanation(
            is_anomaly, factors, activity, movement, confidence
        )
        
        return AnomalyScore(
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_type=anomaly_type if is_anomaly else None,
            contributing_factors=factors,
            explanation=explanation,
            severity=""  # Will be set in __post_init__
        )
    
    @classmethod
    def _classify_anomaly(
        cls,
        signals: Dict,
        bodypart_anomalies: List[str],
        z_score: float,
        movement: float,
        intensity_score: float,
        is_velocity_anomaly: bool = False,
        is_trend_anomaly: bool = False,
        trend_type: str = "stable"
    ) -> Optional[AnomalyType]:
        """Classify the type of anomaly."""
        # Extremely high movement or velocity anomaly = likely tracking error
        if movement > 1000 or (is_velocity_anomaly and movement > 500):
            return AnomalyType.TRACKING_ERROR
        
        # Velocity anomaly with impossible movement
        if is_velocity_anomaly and signals.get("velocity_anomaly", 0) > 0.8:
            return AnomalyType.IMPOSSIBLE_MOVEMENT
        
        # High Z-score = statistical outlier
        if abs(z_score) > 4.0:
            return AnomalyType.STATISTICAL_OUTLIER
        
        # Unexpected body parts moved
        if bodypart_anomalies and signals["bodypart_analysis"] > 0.7:
            return AnomalyType.UNEXPECTED_BODYPART
        
        # Trend anomaly with erratic movement
        if is_trend_anomaly and trend_type in ("erratic", "jerky"):
            return AnomalyType.IMPOSSIBLE_MOVEMENT
        
        # Intensity way out of range
        if intensity_score > 2.5 or intensity_score < 0.2:
            return AnomalyType.ACTIVITY_MISMATCH
        
        return AnomalyType.IMPOSSIBLE_MOVEMENT
    
    @classmethod
    def _generate_explanation(
        cls,
        is_anomaly: bool,
        factors: Dict,
        activity: str,
        movement: float,
        confidence: float
    ) -> str:
        """Generate human-readable explanation."""
        if not is_anomaly:
            return f"Normal movement for {activity}"
        
        # Build explanation from contributing factors
        if factors:
            reasons = [f"{k}: {v}" for k, v in list(factors.items())[:2]]  # Top 2 reasons
            return f"Anomaly ({confidence:.0%}) - " + "; ".join(reasons)
        else:
            return f"Anomaly detected ({confidence:.0%})"
    
    @classmethod
    def _apply_occlusion_penalty(
        cls,
        confidence: float,
        visibility_confidence: float
    ) -> float:
        """
        Apply penalty to confidence when body visibility is low.
        
        This reduces false positives when only part of the body is visible,
        as we have less information to make accurate anomaly judgments.
        
        Args:
            confidence: Raw anomaly confidence score
            visibility_confidence: Body visibility confidence (0.0-1.0)
            
        Returns:
            Adjusted confidence score
        """
        # Don't penalize if visibility is high
        if visibility_confidence >= 0.8:
            return confidence
        
        # Apply proportional penalty
        # At 50% visibility, reduce confidence by ~25%
        # At 20% visibility, reduce confidence by ~50%
        penalty_factor = 0.5 + (visibility_confidence * 0.5)
        
        return confidence * penalty_factor

