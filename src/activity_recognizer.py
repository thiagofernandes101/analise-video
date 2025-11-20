import numpy as np

class ActivityRecognizer:
    def __init__(self):
        pass

    def recognize(self, keypoints, emotion=None):
        """
        Infer activity based on keypoints and emotion.
        keypoints: numpy array of shape (17, 2) or list of (x, y) tuples.
        emotion: string (e.g., 'happy', 'neutral')
        """
        if len(keypoints) == 0:
            return "Unknown"

        # Keypoint indices (COCO)
        # 5: L Shoulder, 6: R Shoulder
        # 9: L Wrist, 10: R Wrist
        # 11: L Hip, 12: R Hip
        # 13: L Knee, 14: R Knee
        # 15: L Ankle, 16: R Ankle

        # Helper to get y-coord safely (assuming 0 is top)
        def get_y(idx):
            if idx < len(keypoints):
                return keypoints[idx][1]
            return 0

        # 1. Detect Posture (Standing vs Sitting)
        # Heuristic: Check vertical distance between Hip and Knee vs Knee and Ankle
        # Or simply check if thigh is horizontal?
        # Better: Check if Knee is significantly below Hip.
        
        l_hip_y = get_y(11)
        r_hip_y = get_y(12)
        l_knee_y = get_y(13)
        r_knee_y = get_y(14)
        l_ankle_y = get_y(15)
        r_ankle_y = get_y(16)
        
        # Average Y positions
        hip_y = (l_hip_y + r_hip_y) / 2
        knee_y = (l_knee_y + r_knee_y) / 2
        ankle_y = (l_ankle_y + r_ankle_y) / 2
        
        posture = "Unknown"
        
        # Check visibility (if y is 0 or very small, it's likely not detected)
        if hip_y > 0 and knee_y > 0:
            # If knee is significantly below hip, likely standing
            # If knee is roughly same level as hip (in Y), likely sitting (or leg raised)
            # But in 2D, sitting usually means thigh is horizontal, so Knee Y ~= Hip Y? 
            # No, in camera view, if sitting on a chair, knees are usually lower than hips but not as much as standing.
            # Actually, simpler check: Aspect ratio of the bounding box? 
            # Let's stick to keypoints.
            
            # Vertical distance
            thigh_len = abs(knee_y - hip_y)
            shin_len = abs(ankle_y - knee_y) if ankle_y > 0 else 0
            
            # If thigh vertical projection is large -> Standing
            # If thigh vertical projection is small -> Sitting (thigh is foreshortened or horizontal)
            
            # Threshold depends on person size. Normalize by torso length?
            # Torso: Shoulder to Hip
            shoulder_y = (get_y(5) + get_y(6)) / 2
            torso_len = abs(hip_y - shoulder_y)
            
            if torso_len > 0:
                ratio = thigh_len / torso_len
                if ratio > 0.5: # Thigh is long vertically
                    posture = "Standing"
                else:
                    posture = "Sitting"
            else:
                posture = "Standing" # Fallback
        else:
             posture = "Standing" # Fallback if legs not seen (e.g. upper body only)

        # 2. Detect Arm Action (Hands Up)
        l_wrist_y = get_y(9)
        r_wrist_y = get_y(10)
        l_shoulder_y = get_y(5)
        r_shoulder_y = get_y(6)
        
        action = ""
        if l_wrist_y > 0 and l_shoulder_y > 0 and l_wrist_y < l_shoulder_y:
            action = "Hands Up"
        elif r_wrist_y > 0 and r_shoulder_y > 0 and r_wrist_y < r_shoulder_y:
            action = "Hands Up"
            
        # 3. Combine with Emotion
        final_label = posture
        if action:
            final_label += f", {action}"
            
        return final_label
