import numpy as np
from collections import deque

class ActivityRecognizer:
    def __init__(self, history_len=30):
        # History: ID -> deque of (keypoints, timestamp/frame_count)
        self.history = {} 
        self.history_len = history_len

    def recognize(self, keypoints, person_id, emotion=None):
        """
        Infer activity based on keypoints, history, and emotion.
        keypoints: numpy array of shape (17, 2)
        person_id: unique identifier for the person
        emotion: string (e.g., 'happy', 'neutral')
        """
        if len(keypoints) == 0:
            return "Unknown"

        # Update history
        self._update_history(person_id, keypoints)

        # Helper to get y-coord safely
        def get_pt(idx):
            if idx < len(keypoints):
                return keypoints[idx]
            return np.array([0, 0])

        # Keypoints
        nose = get_pt(0)
        l_shoulder = get_pt(5)
        r_shoulder = get_pt(6)
        l_wrist = get_pt(9)
        r_wrist = get_pt(10)
        l_hip = get_pt(11)
        r_hip = get_pt(12)
        l_knee = get_pt(13)
        r_knee = get_pt(14)
        l_ankle = get_pt(15)
        r_ankle = get_pt(16)

        # --- 1. Detect Posture (Laying Down vs Standing vs Sitting) ---
        posture = "Standing" # Default

        # Calculate Torso Angle (Verticality)
        shoulder_center = (l_shoulder + r_shoulder) / 2
        hip_center = (l_hip + r_hip) / 2
        
        if shoulder_center[0] > 0 and hip_center[0] > 0:
            dy = hip_center[1] - shoulder_center[1]
            dx = hip_center[0] - shoulder_center[0]
            angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
            
            # If angle is large (> 45), body is likely horizontal -> Laying Down
            if angle > 45:
                posture = "Laying Down"
            else:
                # Check Sitting vs Standing
                # Heuristic: Thigh vertical projection vs Torso length
                knee_y = (l_knee[1] + r_knee[1]) / 2
                hip_y = hip_center[1]
                
                if knee_y > 0 and hip_y > 0:
                    thigh_len = abs(knee_y - hip_y)
                    torso_len = abs(hip_y - shoulder_center[1])
                    
                    if torso_len > 0:
                        ratio = thigh_len / torso_len
                        if ratio < 0.45: # Thigh is short vertically -> Sitting
                            posture = "Sitting"
                        else:
                            posture = "Standing"

        # --- 2. Detect Dynamic Actions (Jumping, Walking, Waving) ---
        action = ""
        
        history = self.history.get(person_id, [])
        # Convert history deque to list for slicing
        history_list = list(history)
        if len(history_list) > 5:
            # A. Jumping
            # Check vertical velocity of hips
            # Get recent hip y positions
            recent_hips = [ (h[11][1] + h[12][1])/2 for h in history_list[-10:] ]
            if len(recent_hips) > 2:
                # Simple variance check or trajectory check
                # If we see a significant Up then Down movement
                y_min = min(recent_hips)
                y_max = max(recent_hips)
                y_range = y_max - y_min
                
                # Normalize by person height (approx torso len)
                person_height = abs(hip_center[1] - shoulder_center[1]) * 2 # rough est
                if person_height > 0 and (y_range / person_height) > 0.2:
                     # Check if it was an upward movement (decreasing Y) followed by down
                     # This is hard to do robustly without ground plane, but high variance implies movement
                     # Let's check if current Y is higher (smaller value) than average
                     avg_y = np.mean(recent_hips)
                     if hip_center[1] < avg_y - (person_height * 0.1): # Currently in air?
                         action = "Jumping"

            # B. Walking
            # Check periodic movement of ankles/knees
            if not action and posture == "Standing":
                # Check ankle x-distance variance (scissor effect)
                recent_l_ankle_x = [h[15][0] for h in history_list[-15:]]
                recent_r_ankle_x = [h[16][0] for h in history_list[-15:]]
                
                # If ankles are moving relative to each other
                if np.var(recent_l_ankle_x) > 50 or np.var(recent_r_ankle_x) > 50: # Threshold needs tuning
                     action = "Walking"

            # C. Waving Hands
            # Check wrist variance high, shoulder variance low
            if not action:
                for wrist, shoulder, name in [(l_wrist, l_shoulder, "Left"), (r_wrist, r_shoulder, "Right")]:
                    if wrist[1] < shoulder[1] and wrist[0] > 0: # Hand above shoulder (roughly)
                        # Check history of this wrist
                        recent_wrist_x = [h[9 if name=="Left" else 10][0] for h in history_list[-10:]]
                        if np.var(recent_wrist_x) > 100: # High variance in X
                            action = "Waving Hand"
                            break
                            
            # D. Waving Head (Bobbing/Shaking)
            if not action:
                recent_nose_x = [h[0][0] for h in history_list[-10:]]
                if np.var(recent_nose_x) > 50: # Threshold?
                     # Ensure body isn't moving as much (walking triggers this too)
                     recent_hip_x = [ (h[11][0] + h[12][0])/2 for h in history_list[-10:] ]
                     if np.var(recent_hip_x) < 20:
                         action = "Waving Head"

        # --- 3. Static Arm Actions ---
        if not action:
             if (l_wrist[1] > 0 and l_shoulder[1] > 0 and l_wrist[1] < l_shoulder[1]) or \
                (r_wrist[1] > 0 and r_shoulder[1] > 0 and r_wrist[1] < r_shoulder[1]):
                 action = "Hands Up"

        # Combine
        final_label = posture
        if action:
            final_label += f", {action}"
            
        return final_label

    def _update_history(self, person_id, keypoints):
        if person_id not in self.history:
            self.history[person_id] = deque(maxlen=self.history_len)
        self.history[person_id].append(keypoints)
