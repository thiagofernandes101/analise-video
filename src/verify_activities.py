import numpy as np
from activity_recognizer import ActivityRecognizer

def create_mock_keypoints(pose="standing", frame_idx=0):
    # Basic skeleton structure (17 points)
    kps = np.zeros((17, 2))
    
    # Common points
    nose = [0, 0]
    l_shoulder = [50, 50]
    r_shoulder = [150, 50]
    l_hip = [60, 150]
    r_hip = [140, 150]
    l_knee = [60, 250]
    r_knee = [140, 250]
    l_ankle = [60, 350]
    r_ankle = [140, 350]
    l_wrist = [40, 150] # Arms down
    r_wrist = [160, 150]

    if pose == "standing":
        pass # Default
        
    elif pose == "sitting":
        # Knees higher (y smaller) or hips lower? 
        # In 2D, sitting often looks like shorter legs if facing camera
        # Or if side view, thigh is horizontal.
        # Let's simulate "short legs" (knees closer to hips in Y)
        l_knee = [60, 180]
        r_knee = [140, 180]
        l_ankle = [60, 250]
        r_ankle = [140, 250]
        
    elif pose == "laying_down":
        # Rotate body 90 degrees
        # Head at 0,0 -> Feet at 300, 0
        nose = [0, 100]
        l_shoulder = [50, 80]
        r_shoulder = [50, 120]
        l_hip = [150, 80]
        r_hip = [150, 120]
        l_knee = [250, 80]
        r_knee = [250, 120]
        l_ankle = [350, 80]
        r_ankle = [350, 120]

    elif pose == "jumping":
        # Move everything up (lower Y)
        offset = -50 if (frame_idx % 10) < 5 else 0 # Jump up and down
        if frame_idx % 20 > 10: # In air
             offset = -100
        
        for p in [nose, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle, l_wrist, r_wrist]:
            p[1] += offset

    elif pose == "walking":
        # Oscillate ankles x
        stride = 30 * np.sin(frame_idx * 0.5)
        l_ankle[0] += stride
        r_ankle[0] -= stride
        
    elif pose == "waving_hand":
        # Right hand up and moving X
        r_wrist[1] = 20 # Above shoulder (50)
        r_wrist[0] = 160 + 40 * np.sin(frame_idx * 1.0)
        
    elif pose == "waving_head":
        # Nose moving X
        nose[0] = 100 + 20 * np.sin(frame_idx * 1.0)

    # Assign
    kps[0] = nose
    kps[5] = l_shoulder
    kps[6] = r_shoulder
    kps[9] = l_wrist
    kps[10] = r_wrist
    kps[11] = l_hip
    kps[12] = r_hip
    kps[13] = l_knee
    kps[14] = r_knee
    kps[15] = l_ankle
    kps[16] = r_ankle
    
    return kps

def test_activity(name, pose_type, duration=30):
    print(f"--- Testing {name} ---")
    recognizer = ActivityRecognizer()
    pid = 1
    
    detected_activities = []
    
    for i in range(duration):
        kps = create_mock_keypoints(pose_type, i)
        label = recognizer.recognize(kps, pid)
        detected_activities.append(label)
        # print(f"Frame {i}: {label}")
        
    # Check if the last few frames detected the activity
    final_label = detected_activities[-1]
    print(f"Final Result: {final_label}")
    return final_label

if __name__ == "__main__":
    test_activity("Standing", "standing")
    test_activity("Sitting", "sitting")
    test_activity("Laying Down", "laying_down")
    test_activity("Jumping", "jumping", 40)
    test_activity("Walking", "walking", 40)
    test_activity("Waving Hand", "waving_hand", 40)
    test_activity("Waving Head", "waving_head", 40)
