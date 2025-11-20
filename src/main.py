import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2 as cv
import threading
import queue
import time
import numpy as np

from config import Config
from detector import PersonDetector
from face_detector import FaceDetector
from emotion_analyzer import EmotionAnalyzer
from visualizer import Visualizer
from activity_recognizer import ActivityRecognizer

def main():
    # 1. Setup
    video_path = Config.get_video_path()
    if not video_path:
        print(f"ERRO: Vídeo não encontrado.")
        return

    print(f"Usando vídeo: {video_path}")
    
    # 2. Initialize Components
    pose_detector = PersonDetector(Config.YOLO_MODEL_SIZE)
    face_detector = FaceDetector() 
    emotion_analyzer = EmotionAnalyzer()
    activity_recognizer = ActivityRecognizer()
    visualizer = Visualizer()
    
    # 3. Warmup (Synchronous)
    print("--- Iniciando Warmup ---")
    pose_detector.warmup()
    face_detector.warmup()
    print("--- Warmup Concluído ---")
    
    video_capture = cv.VideoCapture(video_path)
    frame_count = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("End of video stream.")
                break
                
            frame_count += 1
            
            # --- SYNCHRONOUS PROCESSING (Spatial Data) ---
            # This guarantees that Pose and Face data belong to THIS frame.
            
            # 1. Pose Detection
            results = pose_detector.track(frame)
            pose_data = ([], [], [])
            if results and len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
                keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints.xy is not None else []
                pose_data = (boxes, track_ids, keypoints)
            
            # 2. Face Detection
            face_data = face_detector.detect_faces(frame)
            
            # --- ASYNCHRONOUS PROCESSING (Emotion Analysis) ---
            # Trigger analysis for detected faces.
            # We match faces to Person IDs here to associate the emotion request with an ID.
            
            person_boxes, person_ids, person_keypoints = pose_data
            
            # Map ID -> Emotion
            id_to_emotion = {}
            
            for (fx, fy, fw, fh) in face_data:
                fx_center = fx + fw / 2
                fy_center = fy + fh / 2
                
                matched_id = None
                if len(person_boxes) > 0:
                    for (pbox, pid) in zip(person_boxes, person_ids):
                        px1, py1, px2, py2 = pbox
                        if px1 < fx_center < px2 and py1 < fy_center < py2:
                            matched_id = pid
                            break
                
                if matched_id is not None:
                    # Get current emotion for this ID
                    current_emotion, last_update = emotion_analyzer.get_emotion(matched_id)
                    id_to_emotion[matched_id] = current_emotion
                    
                    # Check if we should analyze
                    if (frame_count - last_update) >= Config.EMOTION_ANALYSIS_INTERVAL:
                         h_img, w_img = frame.shape[:2]
                         fx = max(0, fx)
                         fy = max(0, fy)
                         fw = min(w_img - fx, fw)
                         fh = min(h_img - fy, fh)
                         
                         if fw > 0 and fh > 0:
                             face_img = frame[fy:fy+fh, fx:fx+fw].copy()
                             emotion_analyzer.analyze_async(matched_id, face_img, frame_count)

            # --- ACTIVITY RECOGNITION ---
            # Calculate activity for each person
            activities = {} # ID -> Activity Label
            if len(person_ids) > 0:
                for (pid, kps) in zip(person_ids, person_keypoints):
                    emotion = id_to_emotion.get(pid, None)
                    activity = activity_recognizer.recognize(kps, pid, emotion)
                    activities[pid] = activity

            # --- VISUALIZATION ---
            # Draw everything on the current frame.
            # Pass activities to visualizer
            visualizer.draw_pose(frame, pose_data, activities)
            visualizer.draw_faces_and_emotions(frame, face_data, pose_data, emotion_analyzer)
            visualizer.draw_hud(frame, pose_data, activities, id_to_emotion)
            
            # Cleanup Cache
            if len(person_ids) > 0:
                emotion_analyzer.clean_cache(person_ids)
            
            # Display
            if visualizer.show_frame(frame, frame_count):
                break
                
    finally:
        video_capture.release()
        visualizer.close()
        emotion_analyzer.stop()

if __name__ == "__main__":
    main()