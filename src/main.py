"""
Main application entry point - refactored for Clean Code and SOLID principles.

This module coordinates high-level application flow, delegating specific
responsibilities to specialized components.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2 as cv
from typing import List, Dict

from config import Config
from models.bounding_box import BoundingBox
from models.person_tracking import PersonTracking
from models.face_detection import FaceDetection
from models.emotion_result import EmotionResult
from models.activity_result import ActivityResult
from detectors.person_detector import PersonDetector
from detectors.face_detector import FaceDetector
from analyzers.emotion_analyzer import EmotionAnalyzer
from activity_recognizer import ActivityRecognizer
from visualizer import Visualizer
from services.face_person_matcher import FacePersonMatcher
from services.statistics_tracker import StatisticsTracker
from ui.summary_window import SummaryWindow


class VideoAnalysisApplication:
    """
    Main application orchestrator for video analysis.
    
    Coordinates detection, analysis, and visualization components.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize application with all components.
        
        Args:
            config: Configuration object
        """
        self._config = config
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all analysis components."""
        print("Initializing components...")
        
        self._person_detector = PersonDetector(self._config.model.YOLO_MODEL_NAME)
        self._face_detector = FaceDetector(self._config.model.FACE_DETECTION_BACKEND)
        self._emotion_analyzer = EmotionAnalyzer(self._config)
        self._activity_recognizer = ActivityRecognizer(self._config)
        self._visualizer = Visualizer(self._config)
        self._face_matcher = FacePersonMatcher()
        self._statistics_tracker = StatisticsTracker(self._config)
        
        self._warmup_components()
    
    def _warmup_components(self) -> None:
        """Warm up models with dummy inference."""
        print("--- Starting Warmup ---")
        self._person_detector.warmup()
        self._face_detector.warmup()
        print("--- Warmup Complete ---")
    
    def run(self) -> None:
        """Run the main video analysis loop."""
        video_path = self._config.get_video_path()
        
        if not video_path:
            print("ERROR: Video file not found.")
            return
        
        print(f"Using video: {video_path}")
        
        video_capture = cv.VideoCapture(video_path)
        frame_number = 0
        
        try:
            while True:
                should_stop = self._process_next_frame(
                    video_capture, 
                    frame_number
                )
                
                if should_stop:
                    break
                
                frame_number += 1
        finally:
            self._cleanup(video_capture)
            self._show_summary()
    
    def _process_next_frame(
        self, 
        video_capture: cv.VideoCapture, 
        frame_number: int
    ) -> bool:
        """
        Process a single frame.
        
        Args:
            video_capture: OpenCV video capture object
            frame_number: Current frame number
            
        Returns:
            True if should stop processing, False to continue
        """
        frame_read_successfully, frame = video_capture.read()
        
        if not frame_read_successfully:
            print("End of video stream.")
            return True
        
        # Detect persons and faces
        persons = self._detect_persons(frame)
        faces = self._detect_faces(frame)
        
        # Match faces to persons
        self._assign_faces_to_persons(faces, persons)
        
        # Analyze emotions asynchronously
        emotions = self._analyze_emotions(frame, faces, frame_number)
        
        # Recognize activities
        activities = self._recognize_activities(persons)
        
        # Cleanup
        self._cleanup_caches(persons)
        
        # Visualize
        self._visualizer.render_frame(frame, persons, faces, activities, emotions)
        should_quit = self._visualizer.show_frame(frame, frame_number)
        
        # Track statistics
        self._statistics_tracker.update(
            frame_number=frame_number,
            persons=persons,
            activities=activities,
            emotions=emotions
        )
        
        return should_quit
    
    def _detect_persons(self, frame) -> List[PersonTracking]:
        """
        Detect and track persons in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of PersonTracking objects
        """
        results = self._person_detector.track(frame)
        
        if not results or len(results) == 0:
            return []
        
        persons = []
        result = results[0]
        
        if result.boxes.xyxy is None:
            return []
        
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = (
            result.boxes.id.cpu().numpy().astype(int) 
            if result.boxes.id is not None 
            else []
        )
        keypoints = (
            result.keypoints.xy.cpu().numpy() 
            if result.keypoints.xy is not None 
            else []
        )
        
        if len(track_ids) == 0 or len(keypoints) == 0:
            return []
        
        for box, track_id, kps in zip(boxes, track_ids, keypoints):
            x1, y1, x2, y2 = box
            bounding_box = BoundingBox.from_xyxy(
                int(x1), int(y1), int(x2), int(y2)
            )
            
            person = PersonTracking(
                track_id=int(track_id),
                bounding_box=bounding_box,
                keypoints=kps
            )
            persons.append(person)
        
        return persons
    
    def _detect_faces(self, frame) -> List[FaceDetection]:
        """
        Detect faces in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of FaceDetection objects
        """
        face_data = self._face_detector.detect_faces(frame)
        
        faces = []
        for (face_x, face_y, face_width, face_height) in face_data:
            bounding_box = BoundingBox.from_xywh(
                face_x, face_y, face_width, face_height
            )
            face = FaceDetection(bounding_box=bounding_box)
            faces.append(face)
        
        return faces
    
    def _assign_faces_to_persons(
        self, 
        faces: List[FaceDetection], 
        persons: List[PersonTracking]
    ) -> None:
        """
        Assign faces to persons based on spatial overlap.
        
        Args:
            faces: List of detected faces (modified in-place)
            persons: List of tracked persons
        """
        for face in faces:
            person_id = self._face_matcher.find_person_for_face(face, persons)
            face.person_id = person_id
    
    def _analyze_emotions(
        self,
        frame,
        faces: List[FaceDetection],
        frame_number: int
    ) -> Dict[int, EmotionResult]:
        """
        Analyze emotions for detected faces.
        
        Args:
            frame: Input image frame
            faces: List of detected faces with person assignments
            frame_number: Current frame number
            
        Returns:
            Dict mapping person_id to EmotionResult
        """
        emotions = {}
        
        for face in faces:
            if not face.has_person_assignment:
                continue
            
            person_id = face.person_id
            
            # Get current emotion from cache
            current_emotion = self._emotion_analyzer.get_emotion(person_id)
            emotions[person_id] = current_emotion
            
            # Check if should analyze
            should_analyze = self._should_analyze_emotion(
                current_emotion, 
                frame_number
            )
            
            if should_analyze:
                face_image = self._extract_face_image(frame, face)
                if face_image is not None:
                    self._emotion_analyzer.analyze_async(
                        person_id, 
                        face_image, 
                        frame_number
                    )
        
        return emotions
    
    def _should_analyze_emotion(
        self, 
        current_emotion: EmotionResult, 
        frame_number: int
    ) -> bool:
        """Check if enough frames have passed to analyze again."""
        frames_since_analysis = frame_number - current_emotion.frame_number
        interval = self._config.emotion.ANALYSIS_INTERVAL_FRAMES
        return frames_since_analysis >= interval
    
    def _extract_face_image(self, frame, face: FaceDetection):
        """Extract face region from frame."""
        frame_height, frame_width = frame.shape[:2]
        
        face_x, face_y, face_width, face_height = face.bounding_box.to_xywh_tuple()
        
        # Clamp to frame boundaries
        face_x = max(0, face_x)
        face_y = max(0, face_y)
        face_width = min(frame_width - face_x, face_width)
        face_height = min(frame_height - face_y, face_height)
        
        if face_width <=0 or face_height <= 0:
            return None
        
        return frame[face_y:face_y+face_height, face_x:face_x+face_width].copy()
    
    def _recognize_activities(
        self, 
        persons: List[PersonTracking]
    ) -> Dict[int, ActivityResult]:
        """
        Recognize activities for all persons.
        
        Args:
            persons: List of tracked persons
            
        Returns:
            Dict mapping person_id to ActivityResult
        """
        activities = {}
        
        for person in persons:
            activity = self._activity_recognizer.recognize_activity(person)
            activities[person.track_id] = activity
        
        return activities
    
    def _cleanup_caches(self, persons: List[PersonTracking]) -> None:
        """Clean up caches for persons no longer tracked."""
        active_ids = [person.track_id for person in persons]
        
        self._emotion_analyzer.clean_cache(active_ids)
        self._activity_recognizer.cleanup_inactive_persons(active_ids)
    
    def _cleanup(self, video_capture: cv.VideoCapture) -> None:
        """Clean up resources."""
        video_capture.release()
        self._visualizer.close()
        self._emotion_analyzer.stop()
    
    def _show_summary(self) -> None:
        """Display video analysis summary window."""
        print("\nGenerating summary...")
        statistics = self._statistics_tracker.get_summary()
        
        print(f"Total frames analyzed: {statistics.total_frames}")
        print(f"Persons detected: {statistics.get_person_count()}")
        print(f"Anomalies detected: {statistics.get_anomaly_count()}")
        
        summary_window = SummaryWindow(self._config)
        summary_window.show_and_wait(statistics)


def main():
    """Application entry point."""
    config = Config()
    app = VideoAnalysisApplication(config)
    app.run()


if __name__ == "__main__":
    main()