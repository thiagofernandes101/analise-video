import cv2 as cv
import os
import traceback
import numpy as np

class Visualizer:
    def draw_pose(self, frame, pose_data, activities=None):
        """
        Desenha esqueleto e bounding box da pessoa.
        pose_data: (boxes, track_ids, keypoints)
        activities: dict {track_id: activity_label}
        """
        if pose_data is None:
            return

        boxes, track_ids, keypoints = pose_data
        (h, w) = frame.shape[:2]

        if len(boxes) > 0 and len(track_ids) > 0:
            for (box, track_id, kps) in zip(boxes, track_ids, keypoints):
                (x1, y1, x2, y2) = box.astype("int")
                (x1, y1) = (max(0, x1), max(0, y1))
                (x2, y2) = (min(w - 1, x2), min(h - 1, y2))
                
                # Draw Person BBox (Cyan)
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                
                # Draw ID and Activity
                label = f"ID: {track_id}"
                if activities and track_id in activities:
                    label += f" | {activities[track_id]}"
                
                # Smart Label Positioning
                label_y = y1 - 10
                if label_y < 20: # Too close to top
                    label_y = y1 + 20 # Move inside/below top edge
                
                cv.putText(frame, label, (x1, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw Keypoints
                for (px, py) in kps:
                     if px > 0 and py > 0:
                        cv.circle(frame, (int(px), int(py)), 3, (0, 0, 255), -1)

    def draw_faces_and_emotions(self, frame, face_data, pose_data, emotion_analyzer):
        """
        Desenha bounding box do rosto e emoção.
        Tenta associar rosto a um ID de pessoa para pegar a emoção correta.
        """
        if face_data is None:
            return

        faces = face_data # list of (x, y, w, h)
        (h_frame, w_frame) = frame.shape[:2]
        
        # Unpack pose data for matching
        person_boxes = []
        person_ids = []
        if pose_data:
            person_boxes, person_ids, _ = pose_data

        for (fx, fy, fw, fh) in faces:
            # Draw Face Box (Green)
            cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            
            # Find owner ID
            owner_id = None
            
            # Simple center matching
            fx_center = fx + fw / 2
            fy_center = fy + fh / 2
            
            if len(person_boxes) > 0:
                for (pbox, pid) in zip(person_boxes, person_ids):
                    px1, py1, px2, py2 = pbox
                    if px1 < fx_center < px2 and py1 < fy_center < py2:
                        owner_id = pid
                        break
            
            emotion_text = ""
            if owner_id is not None:
                emotion, _ = emotion_analyzer.get_emotion(owner_id)
                emotion_text = f"{emotion}"
            else:
                emotion_text = "?"

            # Draw Emotion Label
            label = f"{emotion_text}"
            
            # Smart Label Positioning for Face
            label_y = fy - 10
            if label_y < 20:
                label_y = fy + fh + 20 # Move below face if too close to top
            
            cv.putText(frame, label, (fx, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_hud(self, frame, pose_data, activities, id_to_emotion):
        """
        Desenha um painel lateral com informações de todos os IDs rastreados.
        """
        if pose_data is None:
            return
            
        _, track_ids, _ = pose_data
        if len(track_ids) == 0:
            return

        (h, w) = frame.shape[:2]
        
        # HUD Configuration
        panel_w = 300
        panel_h = len(track_ids) * 30 + 40
        panel_x = w - panel_w - 10
        panel_y = 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        alpha = 0.6
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Header
        cv.putText(frame, "Active Tracks", (panel_x + 10, panel_y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # List items
        for i, track_id in enumerate(track_ids):
            y_pos = panel_y + 60 + (i * 30)
            
            activity = activities.get(track_id, "Unknown")
            emotion = id_to_emotion.get(track_id, "N/A")
            
            text = f"ID {track_id}: {activity} | {emotion}"
            cv.putText(frame, text, (panel_x + 10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def show_frame(self, frame, frame_count):
        try:
            if os.environ.get('DISPLAY'):
                cv.imshow('Video Analysis', frame)
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
