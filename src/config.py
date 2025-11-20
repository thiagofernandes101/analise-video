import os

class Config:
    # Environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_FILENAME = 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
    
    # Paths
    POSSIBLE_PATHS = [
        os.path.join(BASE_DIR, '../videos', VIDEO_FILENAME),
        os.path.join('/app/videos', VIDEO_FILENAME)
    ]
    
    # Model
    YOLO_MODEL_SIZE = "yolov8m-pose.pt"
    
    # Analysis
    EMOTION_ANALYSIS_INTERVAL = 15
    EMOTION_CONFIDENCE_THRESHOLD = 40.0
    
    @classmethod
    def get_video_path(cls):
        for path in cls.POSSIBLE_PATHS:
            if os.path.exists(path):
                return path
        return None
