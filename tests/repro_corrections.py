import sys
import os
import numpy as np
from collections import deque
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.video_statistics import PersonStatistics, FrameInfo, MovementSegment
from services.movement_categorizer import MovementCategorizer
from services.statistics_tracker import StatisticsTracker
from config import Config

# Mock classes to support testing StatisticsTracker
class MockPerson:
    def __init__(self, track_id, keypoints):
        self.track_id = track_id
        self.keypoints = keypoints
        
    def get_keypoint_safely(self, idx):
        return self.keypoints[idx]

class MockResult:
    def __init__(self, label="Standing", posture="Standing"):
        self.label = label
        self.posture = posture
        self.action = None
    def get_display_label(self): return self.label

class MockEmotion:
    def __init__(self, emotion):
        self.emotion = emotion

def test_corrections():
    print("Testing Corrections...")
    config = Config()
    # Ensure config has expected attributes
    if not hasattr(config, 'movement_detection'): config.movement_detection = type('obj', (object,), {})
    if not hasattr(config, 'summary'): config.summary = type('obj', (object,), {})
    config.movement_detection.MIN_TRACK_FRAMES = 5
    config.summary.EMOTION_SMOOTHING_WINDOW = 3
    
    tracker = StatisticsTracker(config)
    
    # Test 1: Emotion Smoothing
    print("\nTest 1: Emotion Smoothing")
    # Sequence: Neutral, Neutral, Happy, Neutral -> Should stay Neutral/Happy smooth
    emotions = ["Neutral", "Neutral", "Happy", "Neutral", "Happy", "Happy", "Happy"]
    
    for i, emo in enumerate(emotions):
        person = MockPerson(1, np.zeros((17, 2)))
        tracker.update(i, [person], {1: MockResult()}, {1: MockEmotion(emo)})
        
    stats = tracker._statistics.person_stats[1]
    print(f"Recorded Emotions (Unique): {stats.emotions}")
    # We expect 'Happy' to be recorded only after it becomes dominant or raw fallback works
    # With window 3: [N], [N,N], [N,N,H]->N, [N,H,N]->N, [H,N,H]->H
    
    # Test 2: Short Track Filtering
    print("\nTest 2: Short Track Filtering")
    # Person 2 only appears for 2 frames
    tracker.update(100, [MockPerson(2, np.zeros((17, 2)))], {}, {})
    tracker.update(101, [MockPerson(2, np.zeros((17, 2)))], {}, {})
    
    summary = tracker.get_summary()
    if 2 not in summary.person_stats:
        print("PASS: Short track (Person 2) filtered out.")
    else:
        print(f"FAIL: Person 2 still present with {summary.person_stats[2].frame_count} frames.")
        
    if 1 in summary.person_stats:
        print(f"PASS: Long track (Person 1) retained with {summary.person_stats[1].frame_count} frames.")
    else:
        print("FAIL: Person 1 incorrectly filtered.")

if __name__ == "__main__":
    try:
        test_corrections()
        print("\nTest execution completed.")
    except Exception as e:
        print(f"\nError detected: {e}")
        import traceback
        traceback.print_exc()
