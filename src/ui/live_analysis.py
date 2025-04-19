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
    """Widget for displaying real-time stability gauge with improved visualization."""
    
    def __init__(self):
        """Initialize the stability gauge with professional styling."""
        super().__init__()
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(50)  # Default value
        self.setTextVisible(True)
        self.setFormat("Stability: %v%")
        self.setMinimumHeight(30)
        
        # Apply color styling for professional look
        self.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CFD8DC;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                color: white;
                font-size: 14px;
                padding: 1px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E53935,
                    stop:0.4 #FFB300,
                    stop:0.6 #FFB300,
                    stop:1 #43A047
                );
                border-radius: 5px;
            }
        """)
    
    def update_stability(self, stability_score: float):
        """
        Update the stability gauge with a new score using an improved calculation.
        
        Args:
            stability_score: Raw stability score between 0 and 1
        """
        # Convert to percentage (0-100) with more nuanced scaling
        # This ensures the gauge is more responsive and accurate
        
        # Apply non-linear transformation to better highlight differences
        # in the mid-range which is most relevant for shooting analysis
        if stability_score <= 0.5:
            # Scale lower half to be more sensitive
            percentage = int(40 * (stability_score / 0.5))
        else:
            # Scale upper half to show excellence
            percentage = int(40 + 60 * ((stability_score - 0.5) / 0.5))
        
        # Ensure value is within valid range
        percentage = max(0, min(100, percentage))
        
        # Update the gauge
        self.setValue(percentage)
        
        # Update color based on value ranges
        if percentage < 30:
            self.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #CFD8DC;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #E53935;
                    border-radius: 5px;
                }
            """)
        elif percentage < 70:
            self.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #CFD8DC;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #FFB300;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #CFD8DC;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #43A047;
                    border-radius: 5px;
                }
            """)


class LiveAnalysisWidget(QWidget):
    """
    Widget for real-time shooting analysis with camera feed, metrics, and feedback.
    """
    
    # Signal for when a shot is detected
    shot_detected_signal = pyqtSignal()
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the live analysis widget with improved session flow.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        self.session_id = None
        self.session_active = False
        
        # Initialize components
        self.joint_tracker = JointTracker()
        self.stability_metrics = StabilityMetrics()
        self.fuzzy_feedback = FuzzyFeedback()
        
        # Session shots counter
        self.session_shots = 0
        self.session_scores = []
        
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
        """Initialize the user interface elements with professional styling."""
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
        self.session_label.setStyleSheet("""
            background-color: #E3F2FD;
            padding: 5px 10px;
            border-radius: 4px;
            color: #1565C0;
            font-weight: bold;
        """)
        controls_layout.addWidget(self.session_label)
        
        controls_layout.addStretch()
        
        # Add legend for joint colors
        legend_layout = QHBoxLayout()
        
        # High stability indicator
        high_stability = QLabel("Stable")
        high_stability.setStyleSheet("""
            color: #388E3C;
            font-weight: bold;
            padding-left: 20px;
            background-image: url('');
            background-position: left center;
            background-repeat: no-repeat;
            background-color: transparent;
        """)
        legend_layout.addWidget(high_stability)
        
        # Draw a green circle for high stability indicator
        high_stability_circle = QLabel()
        high_stability_circle.setFixedSize(12, 12)
        high_stability_circle.setStyleSheet("""
            background-color: #4CAF50;
            border-radius: 6px;
        """)
        legend_layout.insertWidget(0, high_stability_circle)
        
        # Medium stability indicator
        medium_stability = QLabel("Medium")
        medium_stability.setStyleSheet("""
            color: #F57C00;
            font-weight: bold;
            padding-left: 20px;
        """)
        legend_layout.addWidget(medium_stability)
        
        # Draw an orange circle for medium stability indicator
        medium_stability_circle = QLabel()
        medium_stability_circle.setFixedSize(12, 12)
        medium_stability_circle.setStyleSheet("""
            background-color: #FFA000;
            border-radius: 6px;
        """)
        legend_layout.insertWidget(2, medium_stability_circle)
        
        # Low stability indicator
        low_stability = QLabel("Unstable")
        low_stability.setStyleSheet("""
            color: #D32F2F;
            font-weight: bold;
            padding-left: 20px;
        """)
        legend_layout.addWidget(low_stability)
        
        # Draw a red circle for low stability indicator
        low_stability_circle = QLabel()
        low_stability_circle.setFixedSize(12, 12)
        low_stability_circle.setStyleSheet("""
            background-color: #F44336;
            border-radius: 6px;
        """)
        legend_layout.insertWidget(4, low_stability_circle)
        
        controls_layout.addLayout(legend_layout)
        
        main_layout.addLayout(controls_layout)
        
        # Main content splitter (camera feed and metrics)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Camera feed
        camera_widget = QWidget()
        camera_layout = QVBoxLayout()
        
        # Add a professional-looking frame around the camera view
        camera_frame = QFrame()
        camera_frame.setFrameShape(QFrame.Shape.StyledPanel)
        camera_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #CFD8DC;
                border-radius: 8px;
                background-color: #263238;
            }
        """)
        camera_frame_layout = QVBoxLayout()
        
        self.camera_view = CameraWidget()
        self.camera_view.setStyleSheet("border: none;")
        camera_frame_layout.addWidget(self.camera_view)
        camera_frame.setLayout(camera_frame_layout)
        
        camera_layout.addWidget(camera_frame)
        
        # Add stability gauge under camera
        gauge_layout = QVBoxLayout()
        gauge_label = QLabel("Overall Stability")
        gauge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gauge_label.setStyleSheet("font-weight: bold; color: #455A64; margin-bottom: 5px;")
        gauge_layout.addWidget(gauge_label)
        
        self.stability_gauge = StabilityGauge()
        gauge_layout.addWidget(self.stability_gauge)
        
        camera_layout.addLayout(gauge_layout)
        
        camera_widget.setLayout(camera_layout)
        main_splitter.addWidget(camera_widget)
        
        # Right side: Metrics and feedback with professional styling
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        
        # Group box for real-time metrics with professional styling
        metrics_group = QGroupBox("Real-Time Stability Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CFD8DC;
                border-radius: 4px;
                margin-top: 1.5ex;
                padding-top: 1ex;
                background-color: #FAFAFA;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #1565C0;
            }
        """)
        metrics_grid = QGridLayout()
        
        # Labels for metrics with better styling
        header_style = "font-weight: bold; color: #455A64; padding: 5px; border-bottom: 1px solid #CFD8DC;"
        metrics_grid.addWidget(QLabel("Joint"), 0, 0)
        metrics_grid.itemAtPosition(0, 0).widget().setStyleSheet(header_style)
        
        metrics_grid.addWidget(QLabel("Sway (mm/s)"), 0, 1)
        metrics_grid.itemAtPosition(0, 1).widget().setStyleSheet(header_style)
        
        metrics_grid.addWidget(QLabel("DevX (px)"), 0, 2)
        metrics_grid.itemAtPosition(0, 2).widget().setStyleSheet(header_style)
        
        metrics_grid.addWidget(QLabel("DevY (px)"), 0, 3)
        metrics_grid.itemAtPosition(0, 3).widget().setStyleSheet(header_style)
        
        # Placeholders for metrics values with consistent styling
        self.metric_labels = {}
        cell_style = "padding: 5px; border-bottom: 1px solid #ECEFF1;"
        
        row = 1
        for joint in ["WRISTS", "ELBOWS", "SHOULDERS", "NOSE", "HIPS"]:
            joint_label = QLabel(joint)
            joint_label.setStyleSheet(f"{cell_style} font-weight: bold;")
            metrics_grid.addWidget(joint_label, row, 0)
            
            sway_label = QLabel("0.00")
            sway_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sway_label.setStyleSheet(cell_style)
            metrics_grid.addWidget(sway_label, row, 1)
            self.metric_labels[f"{joint}_sway"] = sway_label
            
            dev_x_label = QLabel("0.00")
            dev_x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dev_x_label.setStyleSheet(cell_style)
            metrics_grid.addWidget(dev_x_label, row, 2)
            self.metric_labels[f"{joint}_dev_x"] = dev_x_label
            
            dev_y_label = QLabel("0.00")
            dev_y_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dev_y_label.setStyleSheet(cell_style)
            metrics_grid.addWidget(dev_y_label, row, 3)
            self.metric_labels[f"{joint}_dev_y"] = dev_y_label
            
            row += 1
        
        # Follow-through score with special styling
        metrics_grid.addWidget(QLabel("Follow-through:"), row, 0)
        metrics_grid.itemAtPosition(row, 0).widget().setStyleSheet(f"{cell_style} font-weight: bold;")
        
        self.follow_through_label = QLabel("0.00")
        self.follow_through_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.follow_through_label.setStyleSheet(f"{cell_style} font-weight: bold;")
        metrics_grid.addWidget(self.follow_through_label, row, 1, 1, 3)
        
        metrics_group.setLayout(metrics_grid)
        metrics_layout.addWidget(metrics_group)
        
        # Feedback group with professional styling
        feedback_group = QGroupBox("Real-Time Feedback")
        feedback_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CFD8DC;
                border-radius: 4px;
                margin-top: 1.5ex;
                padding-top: 1ex;
                background-color: #FAFAFA;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #1565C0;
            }
        """)
        feedback_layout = QVBoxLayout()
        
        self.feedback_label = QLabel("Start analysis to receive feedback.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            background-color: #E3F2FD; 
            border: 1px solid #BBDEFB; 
            border-radius: 5px; 
            padding: 10px;
            color: #0D47A1;
            min-height: 60px;
        """)
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_layout.addWidget(self.feedback_label)
        
        feedback_group.setLayout(feedback_layout)
        metrics_layout.addWidget(feedback_group)
        
        # Audio threshold control with professional styling
        audio_group = QGroupBox("Shot Detection")
        audio_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CFD8DC;
                border-radius: 4px;
                margin-top: 1.5ex;
                padding-top: 1ex;
                background-color: #FAFAFA;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #1565C0;
            }
        """)
        audio_layout = QHBoxLayout()
        
        audio_layout.addWidget(QLabel("Audio Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(self.audio_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_audio_threshold)
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #CFD8DC;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        audio_layout.addWidget(self.threshold_slider)
        
        self.threshold_value = QLabel(f"{self.audio_threshold:.2f}")
        self.threshold_value.setStyleSheet("""
            min-width: 40px;
            font-weight: bold;
            padding: 2px 5px;
            background-color: #E3F2FD;
            border-radius: 3px;
        """)
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
        """Toggle the analysis (start/stop) with automatic recording."""
        if not self.camera_running:
            if not self.session_id:
                QMessageBox.warning(self, "No Active Session", 
                                "Please create a session first before starting analysis.")
                return
                
            self.start_analysis()
        else:
            self.stop_analysis()
    
    def start_analysis(self):
        """Start real-time analysis with automatic recording."""
        # Start joint tracker
        self.joint_tracker.start()
        
        # Start audio detection
        self.start_audio_detection()
        
        # Start update timer
        self.update_timer.start(33)  # ~30 FPS
        
        # Update UI
        self.start_stop_button.setText("Stop Analysis")
        self.camera_running = True
        self.session_active = True
        
        if self.session_id:
            self.manual_shot_button.setEnabled(True)
        
        # Update main window button if accessible
        main_window = self.window()
        if hasattr(main_window, 'session_button'):
            main_window.session_button.setText("End Session")
    
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
    
    def end_session(self):
        """End the current session and show summary."""
        if not self.session_active:
            return
            
        # Stop analysis if running
        if self.camera_running:
            self.stop_analysis()
        
        self.session_active = False
        
        # Show session summary
        self.show_session_summary()
        
        # Reset session data
        self.session_shots = 0
        self.session_scores = []
        
        # Reset session in main window
        main_window = self.window()
        if hasattr(main_window, 'current_session'):
            main_window.current_session = None
            main_window.session_label.setText("No active session")
        
        if hasattr(main_window, 'session_button'):
            main_window.session_button.setText("New Session")

    def show_session_summary(self):
        """Show a summary popup with session statistics."""
        if not self.session_id:
            return
            
        try:
            # Get session statistics
            stats = self.data_storage.get_session_stats(self.session_id)
            
            # Get session details
            self.cursor = self.data_storage.conn.cursor()
            self.cursor.execute("SELECT name FROM sessions WHERE id = ?", (self.session_id,))
            session = self.cursor.fetchone()
            
            if not session:
                return
                
            # Create summary message
            summary = f"<h2>Session Summary: {session['name']}</h2>"
            summary += "<hr>"
            summary += f"<p><b>Shots Taken:</b> {stats.get('shot_count', 0)}</p>"
            
            if stats.get('shot_count', 0) > 0:
                summary += f"<p><b>Average Score:</b> {stats.get('avg_subjective_score', 0):.1f}</p>"
                summary += f"<p><b>Best Score:</b> {stats.get('max_subjective_score', 0)}</p>"
                summary += f"<p><b>Worst Score:</b> {stats.get('min_subjective_score', 0)}</p>"
            
            # Show the summary in a modal dialog
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Session Complete")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setText(summary)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.setDefaultButton(QMessageBox.StandardButton.Ok)
            
            # Apply professional styling
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    min-width: 400px;
                }
            """)
            
            msg_box.exec()
            
        except Exception as e:
            print(f"Error showing session summary: {e}")

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
        """Handle shot detection (triggered by audio or manually) with session stats tracking."""
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
                # Track session stats
                self.session_shots += 1
                self.session_scores.append(score)
                
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
                
                # Show confirmation
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
    
    def update_analysis(self):
        """Update real-time analysis and UI elements with improved logic."""
        # Get frame with joint tracking
        frame, joint_data, timestamp = self.joint_tracker.get_frame()
        
        if frame is not None:
            # Update camera view with professional framing
            self.camera_view.update_frame(frame)
            
            # Get joint history for stability metrics
            joint_history = self.joint_tracker.get_joint_history()
            
            if joint_history:
                try:
                    # Calculate stability metrics with error handling
                    sway_velocities = self.stability_metrics.calculate_sway_velocity(joint_history)
                    dev_x, dev_y = self.stability_metrics.calculate_postural_stability(joint_history)
                    follow_through = self.stability_metrics.calculate_follow_through_score(joint_history)
                    
                    # Combine metrics for feedback with validation
                    metrics = {
                        'sway_velocity': sway_velocities or {},
                        'dev_x': dev_x or {},
                        'dev_y': dev_y or {},
                        'follow_through_score': max(0.0, min(1.0, follow_through))  # Ensure in valid range
                    }
                    
                    # Generate feedback
                    feedback = self.fuzzy_feedback.generate_feedback(metrics)
                    
                    # Update UI with metrics
                    self.update_metrics_ui(metrics)
                    
                    # Calculate a more accurate stability score based on multiple factors
                    stability_score = self._calculate_overall_stability(metrics)
                    
                    # Update stability gauge with improved calculation
                    self.stability_gauge.update_stability(stability_score)
                    
                    # Update feedback text with professional formatting
                    self.update_feedback_display(feedback['text'])
                    
                except Exception as e:
                    # Log the error and display a user-friendly message
                    import traceback
                    print(f"Error updating analysis: {str(e)}")
                    print(traceback.format_exc())
                    self.feedback_label.setText("Analysis error. Please check camera positioning.")
        
    def _calculate_overall_stability(self, metrics: Dict) -> float:
        """
        Calculate an overall stability score from multiple metrics.
        This provides a more accurate representation than just using the fuzzy feedback score.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Overall stability score between 0 and 1 (higher is better)
        """
        # Extract key metrics
        sway_metrics = metrics.get('sway_velocity', {})
        dev_x_metrics = metrics.get('dev_x', {})
        dev_y_metrics = metrics.get('dev_y', {})
        follow_through = metrics.get('follow_through_score', 0.5)
        
        # Calculate average sway for upper body (most important for shooting)
        upper_body_joints = ['SHOULDERS', 'ELBOWS', 'WRISTS', 'NOSE']
        sway_values = [sway_metrics.get(joint, 0) for joint in upper_body_joints]
        avg_sway = sum(sway_values) / max(1, len(sway_values))
        
        # Calculate average positional deviation
        dev_x_values = [dev_x_metrics.get(joint, 0) for joint in upper_body_joints]
        dev_y_values = [dev_y_metrics.get(joint, 0) for joint in upper_body_joints]
        avg_dev_x = sum(dev_x_values) / max(1, len(dev_x_values))
        avg_dev_y = sum(dev_y_values) / max(1, len(dev_y_values))
        
        # Normalize metrics to 0-1 scale (lower is better for sway and deviation)
        # These thresholds are based on typical shooting stability metrics
        norm_sway = max(0, 1 - (avg_sway / 20.0))
        norm_dev_x = max(0, 1 - (avg_dev_x / 30.0))
        norm_dev_y = max(0, 1 - (avg_dev_y / 30.0))
        
        # Weighted combination of all factors
        # Follow-through and sway are most important for shooting
        stability_score = (
            0.4 * follow_through +   # Follow-through (40% weight)
            0.3 * norm_sway +        # Sway stability (30% weight)
            0.15 * norm_dev_x +      # Horizontal stability (15% weight)
            0.15 * norm_dev_y        # Vertical stability (15% weight)
        )
        
        # Ensure value is in 0-1 range
        return max(0.0, min(1.0, stability_score))

    def update_metrics_ui(self, metrics: Dict):
        """
        Update UI elements with current metrics with improved formatting.
        
        Args:
            metrics: Dictionary of calculated metrics
        """
        # Update joint-specific metrics
        for joint in ["WRISTS", "ELBOWS", "SHOULDERS", "NOSE", "HIPS"]:
            # Sway velocity with color-coding
            sway = metrics['sway_velocity'].get(joint, 0)
            sway_label = self.metric_labels.get(f"{joint}_sway")
            if sway_label:
                sway_label.setText(f"{sway:.2f}")
                
                # Color code based on value (green for good, yellow for moderate, red for high sway)
                if sway < 5.0:  # Low sway (good)
                    sway_label.setStyleSheet("color: #43A047; font-weight: bold;")
                elif sway < 10.0:  # Medium sway
                    sway_label.setStyleSheet("color: #FFB300; font-weight: bold;")
                else:  # High sway (bad)
                    sway_label.setStyleSheet("color: #E53935; font-weight: bold;")
            
            # DevX with color coding
            dev_x = metrics['dev_x'].get(joint, 0)
            dev_x_label = self.metric_labels.get(f"{joint}_dev_x")
            if dev_x_label:
                dev_x_label.setText(f"{dev_x:.2f}")
                
                # Color code
                if dev_x < 10.0:
                    dev_x_label.setStyleSheet("color: #43A047;")
                elif dev_x < 20.0:
                    dev_x_label.setStyleSheet("color: #FFB300;")
                else:
                    dev_x_label.setStyleSheet("color: #E53935;")
            
            # DevY with color coding
            dev_y = metrics['dev_y'].get(joint, 0)
            dev_y_label = self.metric_labels.get(f"{joint}_dev_y")
            if dev_y_label:
                dev_y_label.setText(f"{dev_y:.2f}")
                
                # Color code
                if dev_y < 10.0:
                    dev_y_label.setStyleSheet("color: #43A047;")
                elif dev_y < 20.0:
                    dev_y_label.setStyleSheet("color: #FFB300;")
                else:
                    dev_y_label.setStyleSheet("color: #E53935;")
        
        # Update follow-through score with improved visualization
        follow_through = metrics['follow_through_score']
        if self.follow_through_label:
            self.follow_through_label.setText(f"{follow_through:.2f}")
            
            # Color code based on value (red to green)
            if follow_through < 0.4:
                self.follow_through_label.setStyleSheet("color: #E53935; font-weight: bold;")
            elif follow_through < 0.7:
                self.follow_through_label.setStyleSheet("color: #FFB300; font-weight: bold;")
            else:
                self.follow_through_label.setStyleSheet("color: #43A047; font-weight: bold;")

    def update_feedback_display(self, feedback_text: str):
        """
        Update the feedback display with professional formatting.
        
        Args:
            feedback_text: Feedback text to display
        """
        # Apply professional styling to feedback
        self.feedback_label.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            background-color: #E3F2FD; 
            border: 1px solid #BBDEFB; 
            border-radius: 5px; 
            padding: 10px;
            color: #0D47A1;
        """)
        self.feedback_label.setText(feedback_text)