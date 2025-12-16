import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from services.keypoint_smoother import KeypointSmoother

def test_smoothing():
    print("Testing Kalman Smoothing...")
    smoother = KeypointSmoother()
    track_id = 1
    
    # Simulate a stationary point with jitter
    # True pos: (100, 100)
    true_pos = np.array([100.0, 100.0])
    
    print("\nSimulating stationary point with jitter (Normal Distribution, std=5.0)")
    errors = []
    smoothed_errors = []
    
    # dummy keypoints array (only index 0 matters)
    kpts = np.zeros((17, 2))
    
    for i in range(20):
        noise = np.random.normal(0, 5.0, 2)
        measured_pos = true_pos + noise
        
        kpts[0] = measured_pos
        
        # Apply smoothing
        smoothed_kpts = smoother.smooth(track_id, kpts.copy())
        smoothed_pos = smoothed_kpts[0]
        
        raw_error = np.linalg.norm(measured_pos - true_pos)
        smooth_error = np.linalg.norm(smoothed_pos - true_pos)
        
        errors.append(raw_error)
        smoothed_errors.append(smooth_error)
        
        print(f"Frame {i}: Raw={measured_pos.round(1)}, Smooth={smoothed_pos.round(1)} | Err: {raw_error:.1f} -> {smooth_error:.1f}")

    avg_raw_error = np.mean(errors)
    avg_smooth_error = np.mean(smoothed_errors[5:]) # Skip convergence frames
    
    print(f"\nAverage Error (Raw): {avg_raw_error:.2f}")
    print(f"Average Error (Smoothed, after convergence): {avg_smooth_error:.2f}")
    
    if avg_smooth_error < avg_raw_error:
        print("PASS: Smoothing reduced error.")
    else:
        print("FAIL: Smoothing did not reduce error.")

if __name__ == "__main__":
    test_smoothing()
