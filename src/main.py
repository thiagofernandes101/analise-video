import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2 as cv
from config import Config
from detector import PersonDetector
from emotion_analyzer import EmotionAnalyzer
from visualizer import Visualizer

def main():
    # 1. Setup
    video_path = Config.get_video_path()
    if not video_path:
        print(f"ERRO: Vídeo não encontrado.")
        return

    print(f"Usando vídeo: {video_path}")
    
    # 2. Initialize Components
    detector = PersonDetector(Config.YOLO_MODEL_SIZE)
    emotion_analyzer = EmotionAnalyzer()
    visualizer = Visualizer()
    
    video_capture = cv.VideoCapture(video_path)
    frame_count = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("End of video stream.")
                break
                
            frame_count += 1
            (h, w) = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 3. Detection & Tracking
            results = detector.track(frame)
            
            if results and len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
                keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints.xy is not None else []
            else:
                boxes, track_ids, keypoints = [], [], []

            # 4. Emotion Analysis Logic
            current_tracked_ids = []
            if len(boxes) > 0 and len(track_ids) > 0:
                for (box, track_id) in zip(boxes, track_ids):
                    current_tracked_ids.append(track_id)
                    
                    # Check if analysis is needed
                    last_frame_seen = emotion_analyzer.get_emotion(track_id)[1]
                    should_analyze = (frame_count - last_frame_seen >= Config.EMOTION_ANALYSIS_INTERVAL)
                    
                    if should_analyze:
                        (x1, y1, x2, y2) = box.astype("int")
                        (x1, y1) = (max(0, x1), max(0, y1))
                        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))
                        
                        if x2 > x1 and y2 > y1:
                            face_roi = rgb_frame[y1:y2, x1:x2].copy()
                            emotion_analyzer.analyze_async(track_id, face_roi, frame_count)

            # 5. Visualization
            visualizer.draw_results(frame, boxes, track_ids, keypoints, emotion_analyzer)
            
            # 6. Cleanup
            emotion_analyzer.clean_cache(current_tracked_ids)
            
            # 7. Display
            if visualizer.show_frame(frame, frame_count):
                break
                
    finally:
        video_capture.release()
        visualizer.close()

if __name__ == "__main__":
    main()