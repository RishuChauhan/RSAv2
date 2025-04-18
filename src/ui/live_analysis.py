from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QFrame, QProgressBar, QSlider, QSpinBox,
    QGroupBox, QGridLayout, QComboBox, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QImage, QPixmap

import cv2
import numpy as np
import pyaudio
import struct
import time
import math

from typing import Dict, List, Optional, Tuple

from src.joint_tracking import JointTracker
from src.stability_metrics import StabilityMetrics
from src.fuzzy_feedback import FuzzyFeedback
from src.data_storage import DataStorage

class CameraWidget(QLabel):
    """Widget for displaying camera feed with pose overlay."""
    
    def __init__(self):
        """Initialize the camera widget."""
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Camera feed will appear here")
        self.setStyleSheet("border: 1px solid #ccc;")
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the displayed frame.
        
        Args:
            frame: OpenCV image array
        """
        if frame is None:
            return
        
        # Convert frame to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale image to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio))


class StabilityGauge(QProgressBar):
    """Widget for displaying real-time stability gauge."""
    
    def __init__(self):
        """Initialize the stability gauge."""
        super().__init__()
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(50)  # Default value
        self.setTextVisible(True)
        self.setFormat("Stability: %v%")
        self.setMinimumHeight(30)
        
        # Apply color styling
        self.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff0000,
                    stop:0.5 #ffff00,
                    stop:1 #00ff00
                );
            }
        """)
    
    def update_stability(self, stability_score: float):
        """
        Update the stability gauge with a new score.
        
        Args:
            stability_score: Stability score between 0 and 1
        """
        # Convert to percentage (0-100)
        percentage = int(stability_score * 100)
        self.setValue(percentage)


class LiveAnalysisWidget(QWidget):
    """
    Widget for real-time shooting analysis with camera feed, metrics, and feedback.
    """
    
    # Signal for when a shot is detected
    shot_detected_signal = pyqtSignal()
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the live analysis widget.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        self.session_id = None
        
        # Initialize components
        self.joint_tracker = JointTracker()
        self.stability_metrics = StabilityMetrics()
        self.fuzzy_feedback = FuzzyFeedback()
        
        # Audio detection for shot trigger
        self.audio_threshold = 0.5  # Default threshold (0-1)
        self.setup_audio_detection()
        
        # Initialize UI
        self.init_ui()
        
        # Timer for updating UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_analysis)
        
        # Connect shot detection signal
        self.shot_detected_signal.connect(self.handle_shot_detection)
        
        # Start with components off
        self.camera_running = False
        self.audio_detection_running = False
    
    def init_ui(self):
        """Initialize the user interface elements."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top bar with controls
        controls_layout = QHBoxLayout()
        
        # Start/stop button
        self.start_stop_button = QPushButton("Start Analysis")
        self.start_stop_button.clicked.connect(self.toggle_analysis)
        controls_layout.addWidget(self.start_stop_button)
        
        # Manual shot button
        self.manual_shot_button = QPushButton("Record Shot")
        self.manual_shot_button.clicked.connect(self.manual_shot_detection)
        self.manual_shot_button.setEnabled(False)
        controls_layout.addWidget(self.manual_shot_button)
        
        # Session indicator
        self.session_label = QLabel("No active session")
        controls_layout.addWidget(self.session_label)
        
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)
        
        # Main content splitter (camera feed and metrics)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Camera feed
        camera_widget = QWidget()
        camera_layout = QVBoxLayout()
        
        self.camera_view = CameraWidget()
        camera_layout.addWidget(self.camera_view)
        
        # Add stability gauge under camera
        self.stability_gauge = StabilityGauge()
        camera_layout.addWidget(self.stability_gauge)
        
        camera_widget.setLayout(camera_layout)
        main_splitter.addWidget(camera_widget)
        
        # Right side: Metrics and feedback
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        
        # Group box for real-time metrics
        metrics_group = QGroupBox("Real-Time Stability Metrics")
        metrics_grid = QGridLayout()
        
        # Labels for metrics
        metrics_grid.addWidget(QLabel("Joint"), 0, 0)
        metrics_grid.addWidget(QLabel("Sway (mm/s)"), 0, 1)
        metrics_grid.addWidget(QLabel("DevX (px)"), 0, 2)
        metrics_grid.addWidget(QLabel("DevY (px)"), 0, 3)
        
        # Placeholders for metrics values
        self.metric_labels = {}
        
        row = 1
        for joint in ["WRISTS", "ELBOWS", "SHOULDERS", "NOSE", "HIPS"]:
            metrics_grid.addWidget(QLabel(joint), row, 0)
            
            sway_label = QLabel("0.00")
            metrics_grid.addWidget(sway_label, row, 1)
            self.metric_labels[f"{joint}_sway"] = sway_label
            
            dev_x_label = QLabel("0.00")
            metrics_grid.addWidget(dev_x_label, row, 2)
            self.metric_labels[f"{joint}_dev_x"] = dev_x_label
            
            dev_y_label = QLabel("0.00")
            metrics_grid.addWidget(dev_y_label, row, 3)
            self.metric_labels[f"{joint}_dev_y"] = dev_y_label
            
            row += 1
        
        # Follow-through score
        metrics_grid.addWidget(QLabel("Follow-through:"), row, 0)
        self.follow_through_label = QLabel("0.00")
        metrics_grid.addWidget(self.follow_through_label, row, 1, 1, 3)
        
        metrics_group.setLayout(metrics_grid)
        metrics_layout.addWidget(metrics_group)
        
        # Feedback group
        feedback_group = QGroupBox("Real-Time Feedback")
        feedback_layout = QVBoxLayout()
        
        self.feedback_label = QLabel("Start analysis to receive feedback.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_layout.addWidget(self.feedback_label)
        
        feedback_group.setLayout(feedback_layout)
        metrics_layout.addWidget(feedback_group)
        
        # Audio threshold control
        audio_group = QGroupBox("Shot Detection")
        audio_layout = QHBoxLayout()
        
        audio_layout.addWidget(QLabel("Audio Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(self.audio_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_audio_threshold)
        audio_layout.addWidget(self.threshold_slider)
        
        self.threshold_value = QLabel(f"{self.audio_threshold:.2f}")
        audio_layout.addWidget(self.threshold_value)
        
        audio_group.setLayout(audio_layout)
        metrics_layout.addWidget(audio_group)
        
        metrics_widget.setLayout(metrics_layout)
        main_splitter.addWidget(metrics_widget)
        
        # Set the initial sizes
        main_splitter.setSizes([600, 400])
        
        main_layout.addWidget(main_splitter)
        
        self.setLayout(main_layout)
    
    def set_user(self, user_id: int):
        """
        Set the current user.
        
        Args:
            user_id: ID of the current user
        """
        self.user_id = user_id
        
        # Load user's baseline metrics if available
        baseline = self.data_storage.get_baseline(user_id)
        if baseline:
            self.stability_metrics.baseline_metrics = baseline['metrics']
    
    def set_session(self, session_id: int):
        """
        Set the current session.
        
        Args:
            session_id: ID of the current session
        """
        self.session_id = session_id
        
        # Get session details
        self.cursor = self.data_storage.conn.cursor()
        self.cursor.execute("SELECT name FROM sessions WHERE id = ?", (session_id,))
        session = self.cursor.fetchone()
        
        if session:
            self.session_label.setText(f"Active Session: {session['name']}")
            self.manual_shot_button.setEnabled(True)
        else:
            self.session_label.setText("No active session")
            self.manual_shot_button.setEnabled(False)
    
    def setup_audio_detection(self):
        """Set up audio detection for shot triggering."""
        self.audio = pyaudio.PyAudio()
        
        # Audio parameters
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        
        self.audio_stream = None
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.process_audio)
    
    def start_audio_detection(self):
        """Start audio detection for shot triggering."""
        if self.audio_detection_running:
            return
        
        try:
            self.audio_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.audio_timer.start(50)  # Check audio every 50ms
            self.audio_detection_running = True
            
        except Exception as e:
            QMessageBox.warning(self, "Audio Error", f"Could not start audio detection: {str(e)}")
    
    def stop_audio_detection(self):
        """Stop audio detection."""
        if not self.audio_detection_running:
            return
        
        self.audio_timer.stop()
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        self.audio_detection_running = False
    
    def process_audio(self):
        """Process audio data to detect shot sounds."""
        if not self.audio_stream:
            return
        
        try:
            # Read audio data
            data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
            
            # Convert to int16 array
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            
            # Normalize to 0-1 range (assuming 16-bit audio)
            normalized_rms = rms / 32768.0
            
            # Check if above threshold
            if normalized_rms > self.audio_threshold:
                # Emit signal for shot detection
                self.shot_detected_signal.emit()
                
                # Temporary disable detection to avoid multiple triggers
                self.audio_timer.stop()
                QTimer.singleShot(1000, lambda: self.audio_timer.start(50))
        
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
    
    def update_audio_threshold(self, value: int):
        """
        Update audio threshold from slider.
        
        Args:
            value: Slider value (0-100)
        """
        self.audio_threshold = value / 100.0
        self.threshold_value.setText(f"{self.audio_threshold:.2f}")
    
    def toggle_analysis(self):
        """Toggle the analysis (start/stop)."""
        if not self.camera_running:
            self.start_analysis()
        else:
            self.stop_analysis()
    
    def start_analysis(self):
        """Start real-time analysis."""
        # Start joint tracker
        self.joint_tracker.start()
        
        # Start audio detection
        self.start_audio_detection()
        
        # Start update timer
        self.update_timer.start(33)  # ~30 FPS
        
        # Update UI
        self.start_stop_button.setText("Stop Analysis")
        self.camera_running = True
        
        if self.session_id:
            self.manual_shot_button.setEnabled(True)
    
    def stop_analysis(self):
        """Stop real-time analysis."""
        # Stop update timer
        self.update_timer.stop()
        
        # Stop joint tracker
        self.joint_tracker.stop()
        
        # Stop audio detection
        self.stop_audio_detection()
        
        # Update UI
        self.start_stop_button.setText("Start Analysis")
        self.camera_running = False
        self.manual_shot_button.setEnabled(False)
    
    def update_analysis(self):
        """Update real-time analysis and UI elements."""
        # Get frame with joint tracking
        frame, joint_data, timestamp = self.joint_tracker.get_frame()
        
        if frame is not None:
            # Update camera view
            self.camera_view.update_frame(frame)
            
            # Get joint history for stability metrics
            joint_history = self.joint_tracker.get_joint_history()
            
            if joint_history:
                # Calculate stability metrics
                sway_velocities = self.stability_metrics.calculate_sway_velocity(joint_history)
                dev_x, dev_y = self.stability_metrics.calculate_postural_stability(joint_history)
                follow_through = self.stability_metrics.calculate_follow_through_score(joint_history)
                
                # Combine metrics for feedback
                metrics = {
                    'sway_velocity': sway_velocities,
                    'dev_x': dev_x,
                    'dev_y': dev_y,
                    'follow_through_score': follow_through
                }
                
                # Generate feedback
                feedback = self.fuzzy_feedback.generate_feedback(metrics)
                
                # Update UI with metrics
                self.update_metrics_ui(metrics)
                
                # Update stability gauge
                self.stability_gauge.update_stability(feedback['score'] / 100.0)
                
                # Update feedback text
                self.feedback_label.setText(feedback['text'])
    
    def update_metrics_ui(self, metrics: Dict):
        """
        Update UI elements with current metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
        """
        # Update joint-specific metrics
        for joint in ["WRISTS", "ELBOWS", "SHOULDERS", "NOSE", "HIPS"]:
            # Sway velocity
            sway = metrics['sway_velocity'].get(joint, 0)
            self.metric_labels[f"{joint}_sway"].setText(f"{sway:.2f}")
            
            # DevX
            dev_x = metrics['dev_x'].get(joint, 0)
            self.metric_labels[f"{joint}_dev_x"].setText(f"{dev_x:.2f}")
            
            # DevY
            dev_y = metrics['dev_y'].get(joint, 0)
            self.metric_labels[f"{joint}_dev_y"].setText(f"{dev_y:.2f}")
        
        # Update follow-through score
        follow_through = metrics['follow_through_score']
        self.follow_through_label.setText(f"{follow_through:.2f}")
        
        # Color code based on value (red to green)
        r = int(255 * (1 - follow_through))
        g = int(255 * follow_through)
        self.follow_through_label.setStyleSheet(f"color: rgb({r}, {g}, 0); font-weight: bold;")
    
    def handle_shot_detection(self):
        """Handle shot detection (triggered by audio or manually)."""
        if not self.session_id:
            QMessageBox.warning(self, "No Session", "Please create a session first.")
            return
        
        # Get joint history for the shot
        joint_history = self.joint_tracker.get_joint_history()
        
        if not joint_history:
            QMessageBox.warning(self, "No Data", "No joint tracking data available.")
            return
        
        # Calculate metrics for the shot
        sway_velocities = self.stability_metrics.calculate_sway_velocity(joint_history)
        dev_x, dev_y = self.stability_metrics.calculate_postural_stability(joint_history)
        follow_through = self.stability_metrics.calculate_follow_through_score(joint_history)
        
        # Combine metrics
        metrics = {
            'sway_velocity': sway_velocities,
            'dev_x': dev_x,
            'dev_y': dev_y,
            'follow_through_score': follow_through
        }
        
        # Ask for subjective score
        score, ok = QInputDialog.getInt(
            self, "Shot Recorded", "Enter subjective score (1-10):",
            value=7, min=1, max=10
        )
        
        if ok:
            # Store shot data
            shot_id = self.data_storage.store_shot(self.session_id, metrics, score)
            
            if shot_id > 0:
                # Get user's current baseline
                baseline = self.data_storage.get_baseline(self.user_id)
                current_best_score = baseline['subjective_score'] if baseline else 0
                
                # Update baseline if this is a better shot
                if score > current_best_score:
                    self.stability_metrics.update_baseline(joint_history, score, current_best_score)
                    self.data_storage.update_baseline(
                        self.user_id, 
                        self.stability_metrics.baseline_metrics, 
                        score
                    )
                    QMessageBox.information(self, "New Baseline", 
                                           "New baseline set with this shot!")
                
                QMessageBox.information(self, "Shot Recorded", 
                                       f"Shot recorded with score: {score}")
            else:
                QMessageBox.critical(self, "Error", "Failed to store shot data.")
    
    def manual_shot_detection(self):
        """Manually trigger shot detection."""
        if self.camera_running:
            self.shot_detected_signal.emit()
        else:
            QMessageBox.warning(self, "Analysis Not Running", 
                               "Please start the analysis first.")
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_analysis()
        
        # Clean up audio resources
        if self.audio:
            self.audio.terminate()
        
        event.accept()