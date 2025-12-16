import sys
import os
import numpy as np
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.video_statistics import PersonStatistics, FrameInfo, MovementSegment
from services.movement_categorizer import MovementCategorizer
from config import Config

def create_mock_history(
    activity: str,
    velocity: float,
    frames: int,
    start_frame: int = 0
) -> List[FrameInfo]:
    history = []
    for i in range(frames):
        history.append(FrameInfo(
            frame=start_frame + i,
            emotion="Neutral",
            keypoints=np.zeros((17, 2)), # Dummy keypoints
            activity=activity,
            velocity=velocity,
            is_sudden_movement=False,
            is_tracking_error=False
        ))
    return history

def test_movement_categorization():
    print("Testing Movement Categorization...")
    config = Config()
    # Mock config threshold
    config.summary.ANOMALY_MOVEMENT_THRESHOLD = 50.0
    
    categorizer = MovementCategorizer(config)
    person_stats = PersonStatistics(track_id=1)
    
    # Scenario 1: Sitting with low velocity (Normal)
    print("\nScenario 1: Sitting with low velocity (10.0)")
    history1 = create_mock_history("Sitting", 10.0, 30, start_frame=0)
    person_stats.frame_history.extend(history1)
    
    categorizer.categorize(person_stats)
    
    if person_stats.movement_segments:
        seg = person_stats.movement_segments[0]
        print(f"Segment: {seg.activity}, Anomalies: {seg.anomalies}")
        if not seg.anomalies:
            print("PASS: No anomalies detected.")
        else:
            print("FAIL: Anomalies detected unexpectedly.")
    else:
        print("FAIL: No segments created.")

    # Reset
    person_stats.movement_segments = []
    person_stats.frame_history = []
    
    # Scenario 2: Sitting with high velocity (Anomaly)
    print("\nScenario 2: Sitting with high velocity (60.0)")
    history2 = create_mock_history("Sitting", 60.0, 30, start_frame=0)
    person_stats.frame_history.extend(history2)
    
    categorizer.categorize(person_stats)
    
    if person_stats.movement_segments:
        seg = person_stats.movement_segments[0]
        print(f"Segment: {seg.activity}, Anomalies: {seg.anomalies}")
        if any("High movement" in a for a in seg.anomalies):
            print("PASS: High movement anomaly detected.")
        else:
            print("FAIL: High movement anomaly NOT detected.")
    else:
        print("FAIL: No segments created.")

if __name__ == "__main__":
    try:
        test_movement_categorization()
        print("\nTest execution completed.")
    except Exception as e:
        print(f"\nError detected: {e}")
        import traceback
        traceback.print_exc()
