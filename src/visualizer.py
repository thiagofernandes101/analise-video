import cv2 as cv
import os
import traceback

class Visualizer:
    def draw_results(self, frame, boxes, track_ids, keypoints, emotion_analyzer):
        (h, w) = frame.shape[:2]
        
        if len(boxes) > 0 and len(track_ids) > 0:
            for (box, track_id, kps) in zip(boxes, track_ids, keypoints):
                (x1, y1, x2, y2) = box.astype("int")
                (x1, y1) = (max(0, x1), max(0, y1))
                (x2, y2) = (min(w - 1, x2), min(h - 1, y2))
                
                dominant_emotion = emotion_analyzer.get_emotion(track_id)[0]

                # Draw BBox
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label
                label = f"ID: {track_id} - {dominant_emotion}"
                label_y = y1 - 10 if (y1 - 10) > 0 else (y2 + 20)
                cv.putText(frame, label, (x1, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw Keypoints
                for (px, py) in kps:
                     if px > 0 and py > 0:
                        cv.circle(frame, (int(px), int(py)), 3, (0, 0, 255), -1)

    def show_frame(self, frame, frame_count):
        try:
            if os.environ.get('DISPLAY'):
                cv.imshow('YOLOv8-Pose Tracking & Emotion Analysis (GPU)', frame)
                return cv.waitKey(1) == ord('q')
            else:
                if frame_count % 30 == 0:
                    print(f"Processando frame {frame_count}...")
                return False
        except Exception:
            print("Erro ao exibir frame (cv.imshow):")
            traceback.print_exc()
            os.environ.pop('DISPLAY', None)
            return False
    
    def close(self):
        cv.destroyAllWindows()
