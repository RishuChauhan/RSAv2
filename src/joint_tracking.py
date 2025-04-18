import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

class JointTracker:
    """
    Real-time joint tracking using MediaPipe Pose.
    Extracts 3D joint coordinates at minimum 30 FPS.
    """
    
    # Define key joints we want to track
    TRACKED_JOINTS = [
        # Upper body
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST',
        'NOSE',
        # Lower body
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_ANKLE', 'RIGHT_ANKLE'
    ]
    
    def __init__(self, camera_index: int = 0, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the joint tracker.
        
        Args:
            camera_index: Index of the camera to use
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use most accurate model
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Create video capture object
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure minimum 30 FPS
        
        # Joint coordinate storage
        self.joint_data = {}
        self.joint_history = []  # Store last second of data
        self.max_history_length = 30  # 1 second @ 30 FPS
        
        # Joint index mapping from MediaPipe to our tracked joints
        self.joint_indices = {
            'NOSE': self.mp_pose.PoseLandmark.NOSE.value,
            'LEFT_SHOULDER': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'RIGHT_SHOULDER': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            'LEFT_ELBOW': self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            'RIGHT_ELBOW': self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            'LEFT_WRIST': self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            'RIGHT_WRIST': self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            'LEFT_HIP': self.mp_pose.PoseLandmark.LEFT_HIP.value,
            'RIGHT_HIP': self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            'LEFT_ANKLE': self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'RIGHT_ANKLE': self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        }
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cap.release()
        
    def start(self):
        """Start the camera if it's not already running."""
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            
    def stop(self):
        """Stop the camera and release resources."""
        self.cap.release()
    
    def get_frame(self) -> Tuple[np.ndarray, Dict, float]:
        """
        Get a frame from the camera and process joint positions.
        
        Returns:
            Tuple containing:
            - The processed frame with pose overlay
            - Dictionary of joint positions
            - Timestamp of the frame
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, {}, 0
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        timestamp = time.time()
        results = self.pose.process(frame_rgb)
        
        # Extract joint positions
        joint_data = {}
        if results.pose_landmarks:
            for joint_name in self.TRACKED_JOINTS:
                idx = self.joint_indices[joint_name]
                landmark = results.pose_landmarks.landmark[idx]
                
                # Get 3D coordinates (x, y, z)
                # Note: x and y are normalized to [0.0, 1.0], we multiply by frame dimensions
                joint_data[joint_name] = {
                    'x': landmark.x * frame.shape[1],
                    'y': landmark.y * frame.shape[0],
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            # Store the joint data with timestamp
            self.joint_data = {
                'timestamp': timestamp,
                'joints': joint_data
            }
            
            # Add to history and maintain max length
            self.joint_history.append(self.joint_data)
            if len(self.joint_history) > self.max_history_length:
                self.joint_history.pop(0)
            
            # Draw the pose annotation on the image
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return frame, joint_data, timestamp
    
    def get_joint_history(self) -> List[Dict]:
        """Get the joint history for the last second."""
        return self.joint_history
    
    def set_history_length(self, length: int):
        """Set the maximum history length."""
        self.max_history_length = length
        # Trim history if needed
        while len(self.joint_history) > self.max_history_length:
            self.joint_history.pop(0)
            
    def get_stability_window(self, window_seconds: float = 1.0) -> List[Dict]:
        """
        Get a specific time window of joint data for stability calculations.
        
        Args:
            window_seconds: Window size in seconds
            
        Returns:
            List of joint data dictionaries within the window
        """
        if not self.joint_history:
            return []
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Filter history to get only data within the time window
        return [data for data in self.joint_history 
                if data['timestamp'] >= window_start]