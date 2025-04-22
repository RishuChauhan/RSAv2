import cv2
import mediapipe as mp
import numpy as np
import math 
import time
from typing import Dict, List, Tuple, Optional

class JointTracker:
    """
    Real-time joint tracking using MediaPipe Pose with dynamic stability-based coloring.
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
        Initialize the joint tracker with dynamic coloring.
        
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

        self.camera_index = camera_index

        # Create video capture object
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure minimum 30 FPS
        
        # Joint coordinate storage
        self.joint_data = {}
        self.joint_history = []  # Store last second of data
        self.max_history_length = 30  # 1 second @ 30 FPS
        
        # Joint stability metrics for coloring
        self.joint_stability = {}  # Store stability scores for each joint
        
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
        
        # Create custom drawing specs for different stability levels
        self.drawing_specs = {
            'high_stability': self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Green for high stability
                thickness=2,
                circle_radius=6
            ),
            'medium_stability': self.mp_drawing.DrawingSpec(
                color=(255, 165, 0),  # Orange for medium stability
                thickness=2,
                circle_radius=6
            ),
            'low_stability': self.mp_drawing.DrawingSpec(
                color=(255, 0, 0),  # Red for low stability
                thickness=2,
                circle_radius=6
            ),
            'connections': self.mp_drawing.DrawingSpec(
                color=(213, 213, 213),  # Light gray for connections
                thickness=2
            )
        }
    
    def get_frame(self) -> Tuple[np.ndarray, Dict, float]:
        """
        Get a frame from the camera and process joint positions with stability-based coloring.
        
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
            
            # Calculate joint stability if we have enough history
            if len(self.joint_history) > 5:
                self._calculate_joint_stability()
                
            # Draw the pose annotation with custom stability-based colors
            self._draw_custom_pose(frame, results.pose_landmarks)
        
        return frame, joint_data, timestamp
    
    def _calculate_joint_stability(self):
        """Calculate stability score for each tracked joint based on recent history."""
        # Need at least a few frames to calculate stability
        if len(self.joint_history) < 5:
            return
            
        # Calculate stability score for each joint
        stability_scores = {}
        
        for joint_name in self.TRACKED_JOINTS:
            # Extract position history for this joint
            positions = []
            
            for data in self.joint_history:
                if 'joints' in data and joint_name in data['joints']:
                    pos = data['joints'][joint_name]
                    positions.append((pos['x'], pos['y'], pos['z']))
            
            # Need enough positions to calculate stability
            if len(positions) < 5:
                stability_scores[joint_name] = 0.0
                continue
                
            # Calculate average movement between frames
            movement = 0.0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dz = positions[i][2] - positions[i-1][2]
                
                # Euclidean distance
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                movement += dist
            
            # Average movement (lower is more stable)
            avg_movement = movement / (len(positions) - 1)
            
            # Convert to stability score (0-1 range, higher is more stable)
            # Apply threshold: below 5 is high stability, above 20 is low stability
            if avg_movement < 5.0:
                stability = 1.0  # High stability
            elif avg_movement > 20.0:
                stability = 0.0  # Low stability
            else:
                # Linear scale between 5-20
                stability = 1.0 - ((avg_movement - 5.0) / 15.0)
            
            stability_scores[joint_name] = stability
        
        # Update joint stability scores
        self.joint_stability = stability_scores
    
    def _draw_custom_pose(self, frame, landmarks):
        """
        Draw pose landmarks with colors based on stability.
        
        Args:
            frame: The image to draw on
            landmarks: MediaPipe pose landmarks
        """
        if not landmarks:
            return
        
        # Draw connections first (in background)
        self.mp_drawing.draw_landmarks(
            frame, 
            landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=None,  # We'll draw landmarks manually
            connection_drawing_spec=self.drawing_specs['connections']
        )
        
        # Draw each landmark with stability-based color
        h, w, _ = frame.shape
        
        for joint_name, idx in self.joint_indices.items():
            # Skip joints we don't track
            if joint_name not in self.TRACKED_JOINTS:
                continue
                
            landmark = landmarks.landmark[idx]
            
            # Get pixel coordinates
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Get stability score (default to medium if not calculated yet)
            stability = self.joint_stability.get(joint_name, 0.5)
            
            # Determine drawing spec based on stability
            if stability > 0.7:  # High stability
                spec = self.drawing_specs['high_stability']
            elif stability > 0.3:  # Medium stability
                spec = self.drawing_specs['medium_stability']
            else:  # Low stability
                spec = self.drawing_specs['low_stability']
            
            # Draw custom circle
            cv2.circle(
                frame, 
                (cx, cy), 
                spec.circle_radius,
                spec.color, 
                spec.thickness
            )

    def start(self):
        """Start the tracking process."""
        # Initialize or reset the video capture if needed
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reset tracking data
        self.joint_data = {}
        self.joint_history = []
        self.joint_stability = {}

    def stop(self):
        """Stop the tracking process."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def get_joint_history(self):
        """
        Get the history of joint positions.
        
        Returns:
            List of joint data dictionaries
        """
        return self.joint_history