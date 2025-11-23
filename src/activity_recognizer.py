"""
Activity recognizer - refactored to orchestrate specialized services.

This class now delegates to specialized services following Single Responsibility Principle.
"""
from models.person_tracking import PersonTracking
from models.activity_result import ActivityResult
from services.keypoint_history_manager import KeypointHistoryManager
from services.posture_detector import PostureDetector
from services.action_detector import ActionDetector
from config import Config


class ActivityRecognizer:
    """
    Orchestrates activity recognition by delegating to specialized detectors.
    
    Follows Open/Closed Principle - can extend with new detectors without modification.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize activity recognizer with its dependencies.
        
        Args:
            config: Configuration object
        """
        history_length = config.activity.HISTORY_LENGTH_FRAMES
        
        self._history_manager = KeypointHistoryManager(history_length)
        self._posture_detector = PostureDetector(config)
        self._action_detector = ActionDetector(self._history_manager, config)
        self._config = config
    
    def recognize_activity(self, person: PersonTracking) -> ActivityResult:
        """
        Recognize activity for a person.
        
        Args:
            person: PersonTracking data with keypoints
            
        Returns:
            ActivityResult with detected posture and action
        """
        # Update history for temporal analysis
        self._history_manager.update_history(person.track_id, person.keypoints)
        
        # Detect posture (static)
        posture = self._posture_detector.detect_posture(person)
        
        # Detect action (dynamic)
        action = self._action_detector.detect_action(person, posture)
        
        return ActivityResult(posture=posture, action=action)
    
    def cleanup_inactive_persons(self, active_person_ids: list[int]) -> None:
        """
        Clean up history for persons no longer being tracked.
        
        Args:
            active_person_ids: List of currently active person IDs
        """
        self._history_manager.cleanup_inactive_persons(active_person_ids)
