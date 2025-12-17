"""
Action Classifier Service.

Uses an ST-GCN model (or fallback) to classify actions from skeleton sequences.
"""
import torch
import numpy as np
from typing import List, Optional, Dict, Deque
from collections import deque
import logging
import os

class ActionClassifier:
    """
    Classifies actions using a deep learning model (ST-GCN).
    """
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Buffer config
        # ST-GCN official weights use OpenPose format (18 joints)
        self.window_size = 60 
        self.num_joints = 18  # OpenPose format (not COCO's 17)
        self.channels = 3     # x, y, confidence
        
        # Inference control
        self.inference_interval = 15 # Run every 15 frames (Mitigation)
        
        # Model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Standing", "Walking", "Sitting", "Laying", "Jump", "Fall", "Punch", "Kick"] # Example classes
        
        self._load_model()
        
    def _load_model(self):
        """Load the model weights."""
        model_path = "st_gcn_weights.pt"
        class_file = "src/models/kinetics_400_classes.txt"
        
        # Load classes from file if available
        if os.path.exists(class_file):
            try:
                with open(class_file, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                self.logger.info(f"Loaded {len(self.classes)} action classes from {class_file}")
            except Exception as e:
                self.logger.error(f"Failed to load class file: {e}")
                self.classes = [f"Class {i}" for i in range(400)]
        else:
             # Fallback
             self.classes = [f"Class {i}" for i in range(400)]
        
        try:
            from models.st_gcn import STGCN
            
            # Standard config for Kinetics-400 with OpenPose (18 joints)
            graph_args = {'strategy': 'spatial'}
            self.model = STGCN(
                in_channels=3,  # x, y, confidence
                num_class=400,
                graph_args=graph_args,
                edge_importance_weighting=True
            )
            
            if torch.cuda.is_available() and os.path.exists(model_path):
                # Load weights
                self.model.load_state_dict(torch.load(model_path))
                self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"Deep Action Classifier loaded successfully from {model_path}")
                
            elif os.path.exists(model_path):
                 # CPU load
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                self.logger.info(f"Deep Action Classifier loaded (CPU) from {model_path}")
            else:
                 self.logger.warning(f"Weights file {model_path} not found. Running in PLACEHOLDER mode.")
                 self.model = None

        except Exception as e:
            self.logger.warning(f"Failed to load Action Classifier model: {e}. Using fallback.")
            self.model = None


    def _coco_to_openpose(self, coco_keypoints: np.ndarray) -> np.ndarray:
        """
        Convert COCO format (17 joints) to OpenPose format (18 joints).
        
        COCO: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
              5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
              9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
              13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle
              
        OpenPose: 0-nose, 1-neck, 2-right_shoulder, 3-right_elbow, 4-right_wrist,
                  5-left_shoulder, 6-left_elbow, 7-left_wrist, 8-mid_hip, 9-right_hip,
                  10-right_knee, 11-right_ankle, 12-left_hip, 13-left_knee, 
                  14-left_ankle, 15-right_eye, 16-left_eye, 17-right_ear
        """
        openpose = np.zeros((18, 3))
        
        # Direct mappings
        openpose[0] = [coco_keypoints[0, 0], coco_keypoints[0, 1], 1.0]  # nose
        openpose[15] = [coco_keypoints[2, 0], coco_keypoints[2, 1], 1.0]  # right_eye
        openpose[16] = [coco_keypoints[1, 0], coco_keypoints[1, 1], 1.0]  # left_eye
        openpose[17] = [coco_keypoints[4, 0], coco_keypoints[4, 1], 1.0]  # right_ear
        
        # Shoulders (swapped order in OpenPose)
        openpose[2] = [coco_keypoints[6, 0], coco_keypoints[6, 1], 1.0]  # right_shoulder
        openpose[5] = [coco_keypoints[5, 0], coco_keypoints[5, 1], 1.0]  # left_shoulder
        
        # Arms
        openpose[3] = [coco_keypoints[8, 0], coco_keypoints[8, 1], 1.0]  # right_elbow
        openpose[4] = [coco_keypoints[10, 0], coco_keypoints[10, 1], 1.0]  # right_wrist
        openpose[6] = [coco_keypoints[7, 0], coco_keypoints[7, 1], 1.0]  # left_elbow
        openpose[7] = [coco_keypoints[9, 0], coco_keypoints[9, 1], 1.0]  # left_wrist
        
        # Hips and legs
        openpose[9] = [coco_keypoints[12, 0], coco_keypoints[12, 1], 1.0]  # right_hip
        openpose[10] = [coco_keypoints[14, 0], coco_keypoints[14, 1], 1.0]  # right_knee
        openpose[11] = [coco_keypoints[16, 0], coco_keypoints[16, 1], 1.0]  # right_ankle
        openpose[12] = [coco_keypoints[11, 0], coco_keypoints[11, 1], 1.0]  # left_hip
        openpose[13] = [coco_keypoints[13, 0], coco_keypoints[13, 1], 1.0]  # left_knee
        openpose[14] = [coco_keypoints[15, 0], coco_keypoints[15, 1], 1.0]  # left_ankle
        
        # Computed joints (neck and mid_hip)
        # Neck = average of shoulders
        if coco_keypoints[5, 0] > 0 and coco_keypoints[6, 0] > 0:
            openpose[1] = [
                (coco_keypoints[5, 0] + coco_keypoints[6, 0]) / 2,
                (coco_keypoints[5, 1] + coco_keypoints[6, 1]) / 2,
                1.0
            ]
        
        # Mid hip = average of hips
        if coco_keypoints[11, 0] > 0 and coco_keypoints[12, 0] > 0:
            openpose[8] = [
                (coco_keypoints[11, 0] + coco_keypoints[12, 0]) / 2,
                (coco_keypoints[11, 1] + coco_keypoints[12, 1]) / 2,
                1.0
            ]
            
        return openpose

    def predict(self, keypoint_buffer: np.ndarray) -> str:
        """
        Run inference on a sequence of keypoints.
        
        Args:
            keypoint_buffer: Shape (Frames, Joints=17, 2) - COCO format
            
        Returns:
            Predicted action label
        """
        if self.model is None:
            return "Unknown (No Model)"
            
        try:
            with torch.no_grad():
                # Convert COCO (17, 2) to OpenPose (18, 3) for each frame
                openpose_buffer = np.zeros((len(keypoint_buffer), 18, 3))
                for i, frame_kpts in enumerate(keypoint_buffer):
                    openpose_buffer[i] = self._coco_to_openpose(frame_kpts)
                
                # Prepare input: (N, C, T, V, M)
                # N=Batch, C=Channels(3), T=Frames, V=Vertices(18), M=Persons
                data = torch.tensor(openpose_buffer, dtype=torch.float32)
                data = data.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)  # [1, 3, T, 18, 1]
                data = data.to(self.device)
                
                output = self.model(data)
                pred_idx = output.argmax(dim=1).item()
                return self.classes[pred_idx]
                
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return "Error"

    def is_tracking_buffer_ready(self, buffer_len: int) -> bool:
        """Check if buffer has enough frames for inference."""
        return buffer_len >= self.window_size
