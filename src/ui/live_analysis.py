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
import os

import json

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

        self.gauge_history = []
    
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

        # Add a recording toggle button (checkbox style)
        self.record_enabled = QPushButton("Record")
        self.record_enabled.setCheckable(True)  # Make it a toggle button
        self.record_enabled.setToolTip("Toggle recording - will record when analysis is running")
        self.record_enabled.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                border: 1px solid #CFD8DC;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #E53935;
                color: white;
                font-weight: bold;
            }
        """)
        self.record_enabled.clicked.connect(self.toggle_recording_state)
        controls_layout.addWidget(self.record_enabled)
        
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
        high_stability_circle = QLabel()
        high_stability_circle.setFixedSize(12, 12)
        high_stability_circle.setStyleSheet("""
            background-color: #4CAF50;
            border-radius: 6px;
        """)
        legend_layout.addWidget(high_stability_circle)
        
        high_stability = QLabel("Stable")
        high_stability.setStyleSheet("""
            color: #388E3C;
            font-weight: bold;
            padding-left: 5px;
        """)
        legend_layout.addWidget(high_stability)
        
        # Medium stability indicator
        medium_stability_circle = QLabel()
        medium_stability_circle.setFixedSize(12, 12)
        medium_stability_circle.setStyleSheet("""
            background-color: #FFA000;
            border-radius: 6px;
        """)
        legend_layout.addWidget(medium_stability_circle)
        
        medium_stability = QLabel("Medium")
        medium_stability.setStyleSheet("""
            color: #F57C00;
            font-weight: bold;
            padding-left: 5px;
        """)
        legend_layout.addWidget(medium_stability)
        
        # Low stability indicator
        low_stability_circle = QLabel()
        low_stability_circle.setFixedSize(12, 12)
        low_stability_circle.setStyleSheet("""
            background-color: #F44336;
            border-radius: 6px;
        """)
        legend_layout.addWidget(low_stability_circle)
        
        low_stability = QLabel("Unstable")
        low_stability.setStyleSheet("""
            color: #D32F2F;
            font-weight: bold;
            padding-left: 5px;
        """)
        legend_layout.addWidget(low_stability)
        
        controls_layout.addLayout(legend_layout)
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
        
        # Main content splitter (camera feed and metrics)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Camera feed
        camera_widget = QWidget()
        camera_layout = QVBoxLayout()
        
        # Create recording indicator
        self.recording_indicator = QLabel("● REC")
        self.recording_indicator.setStyleSheet("""
            color: #E53935;
            font-weight: bold;
            padding: 5px;
            border-radius: 3px;
            background-color: rgba(255, 255, 255, 0.7);
        """)
        self.recording_indicator.setVisible(False)
        
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
        
        # Use QHBoxLayout instead of QVBoxLayout for better positioning of the recording indicator
        camera_frame_layout = QHBoxLayout()
        
        # Create a wrapper widget for the camera and recording indicator
        camera_wrapper = QWidget()
        camera_wrapper_layout = QVBoxLayout()
        camera_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add camera view to the wrapper
        self.camera_view = CameraWidget()
        self.camera_view.setStyleSheet("border: none;")
        camera_wrapper_layout.addWidget(self.camera_view)
        
        # Add recording indicator to the wrapper layout
        camera_wrapper_layout.addWidget(self.recording_indicator, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        camera_wrapper.setLayout(camera_wrapper_layout)
        
        # Add wrapper to the frame layout
        camera_frame_layout.addWidget(camera_wrapper)
        camera_frame.setLayout(camera_frame_layout)
        
        # Add frame to camera layout
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
                # Record the shot time
                self.last_shot_time = time.time()
                
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
        """Toggle the analysis with improved error handling."""
        try:
            if not self.camera_running:
                # Check for session more thoroughly
                if not hasattr(self, 'session_id') or not self.session_id:
                    QMessageBox.warning(self, "No Active Session", 
                                    "Please create a session first before starting analysis.")
                    return
                    
                try:
                    self.start_analysis()
                except Exception as e:
                    print(f"Error starting analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    QMessageBox.critical(self, "Error", f"Failed to start analysis: {str(e)}")
            else:
                try:
                    self.stop_analysis()
                except Exception as e:
                    print(f"Error stopping analysis: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to stop analysis: {str(e)}")
        except Exception as e:
            print(f"Error in toggle_analysis: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
    
    def end_session(self):
        """End the current session with improved error handling."""
        try:
            if not self.session_active:
                return
            
            # Set flag to prevent duplicate messages
            self.ending_session = True
            
            # Stop analysis if running
            if self.camera_running:
                self.stop_analysis()
            
            # Stop and save recording if active
            if hasattr(self, 'record_enabled') and self.record_enabled.isChecked() and hasattr(self, 'recording_metadata'):
                self.stop_recording()
            
            # Reset recording toggle
            if hasattr(self, 'record_enabled'):
                self.record_enabled.setChecked(False)
            
            # Show session summary
            self.show_session_summary()
            
            # Reset session data
            self.session_active = False
            self.session_shots = 0
            self.session_scores = []
            self.ending_session = False
            
            # Update main window with proper error handling
            try:
                main_window = self.window()
                if hasattr(main_window, 'current_session'):
                    main_window.current_session = None
                    main_window.session_label.setText("No active session")
                
                if hasattr(main_window, 'session_button'):
                    main_window.session_button.setText("New Session")
            except Exception as e:
                print(f"Error updating main window: {e}")
                
        except Exception as e:
            print(f"Error ending session: {e}")
            # Reset essential states even if there was an error
            self.session_active = False
            self.ending_session = False

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
        """
        Handle shot detection (triggered by audio or manually) with improved follow-through calculation.
        Uses a two-phase approach to ensure post-shot frames are collected for follow-through analysis.
        """
        if not self.session_id:
            QMessageBox.warning(self, "No Session", "Please create a session first.")
            return
        
        # Get joint history for the shot
        joint_history = self.joint_tracker.get_joint_history()
        
        if not joint_history or len(joint_history) == 0:
            QMessageBox.warning(self, "No Data", "No joint tracking data available.")
            return
        
        # IMPORTANT: Use the timestamp of the latest frame as the shot time
        # This ensures the shot time is properly aligned with your tracking timestamps
        latest_frame = joint_history[-1]
        shot_timestamp = latest_frame['timestamp']
        self.last_shot_time = shot_timestamp
        
        print(f"Shot detected at time: {shot_timestamp}")
        print(f"Joint history length: {len(joint_history)}")
        
        if len(joint_history) > 0:
            first_ts = joint_history[0].get('timestamp', 0)
            last_ts = joint_history[-1].get('timestamp', 0)
            print(f"History time range: {first_ts:.3f} to {last_ts:.3f} (span: {last_ts - first_ts:.3f}s)")
        
        # Calculate metrics for the shot
        sway_velocities = self.stability_metrics.calculate_sway_velocity(joint_history)
        dev_x, dev_y = self.stability_metrics.calculate_postural_stability(joint_history)
        
        # Store the initial joint positions at shot time
        current_joint_positions = {}
        if 'joints' in latest_frame:
            current_joint_positions = latest_frame['joints']
            print(f"Captured positions for joints: {list(current_joint_positions.keys())}")
        
        # Store metrics and information needed for delayed processing
        self.pending_shot_data = {
            'timestamp': shot_timestamp,
            'initial_joint_history': joint_history.copy(),
            'sway_velocities': sway_velocities,
            'dev_x': dev_x,
            'dev_y': dev_y,
            'joint_positions': current_joint_positions
        }
        
        # Show feedback to the user about follow-through collection
        self.feedback_label.setText("Shot detected! Collecting follow-through data...")
        
        # Wait to collect post-shot frames (1.5 seconds should be enough to get frames for the 1.0s window)
        QTimer.singleShot(1500, self.complete_shot_processing)
        
        # Change the button state to indicate waiting
        self.manual_shot_button.setEnabled(False)
        self.manual_shot_button.setText("Processing...")
    
    def manual_shot_detection(self):
        """Manually trigger shot detection."""
        if self.camera_running:
            # Record the current time as the shot time
            self.last_shot_time = time.time()
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

                    # For real-time analysis, we should NOT try to calculate follow-through
                    # since we haven't had a shot yet
                    follow_through = 0.0  # Default value during normal tracking
                    
                    # Only show a follow-through score if we're in post-shot mode
                    if hasattr(self, 'last_shot_time') and self.last_shot_time:
                        # Only calculate follow-through if we're within 3 seconds after the shot
                        time_since_shot = time.time() - self.last_shot_time
                        if time_since_shot < 3.0:
                            # Use the actual shot time for follow-through calculation
                            follow_through = self.stability_metrics.calculate_follow_through_score(
                                joint_history, 
                                shot_time=self.last_shot_time,
                                post_window=1.0
                            )
                    
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
        Calculate an overall stability score from sway velocity and positional deviation metrics.
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
        
        # Weighted combination of factors - rebalanced without follow-through
        # Sway is most important for shooting stability
        stability_score = (
            0.6 * norm_sway +        # Sway stability (60% weight)
            0.2 * norm_dev_x +       # Horizontal stability (20% weight)
            0.2 * norm_dev_y         # Vertical stability (20% weight)
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

    def toggle_recording(self):
        """Toggle recording state."""
        if not hasattr(self, 'is_recording') or not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording the current session with metrics data collection."""
        if not self.session_id:
            return
        
        # Get session details
        self.cursor = self.data_storage.conn.cursor()
        self.cursor.execute("SELECT name FROM sessions WHERE id = ?", (self.session_id,))
        session = self.cursor.fetchone()
        
        if not session:
            return
        
        # Ensure recordings directory exists
        self.recordings_dir = "data/recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Create user-specific recordings directory
        self.user_recordings_dir = os.path.join(self.recordings_dir, f"user_{self.user_id}")
        os.makedirs(self.user_recordings_dir, exist_ok=True)
        
        # Create filename with timestamp if not already recording
        if not hasattr(self, 'recording_metadata'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.video_filename = f"session_{self.session_id}_{timestamp}.mp4"
            self.video_path = os.path.join(self.user_recordings_dir, self.video_filename)
            
            # Get video properties
            width = int(self.joint_tracker.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.joint_tracker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.joint_tracker.cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
            
            # Create metadata with properly initialized metrics array
            self.recording_metadata = {
                'user_id': self.user_id,
                'session_id': self.session_id,
                'session_name': session['name'],
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'video_file': self.video_filename,
                'width': width,
                'height': height,
                'fps': fps,
                'metrics': [],  # Initialize empty array for metrics
                'shots': [],    # Initialize empty array for shot markers
                'duration': 0,
                'paused_time': 0  # Track cumulative paused time
            }
            
            # Initialize recording start time
            self.recording_start_time = time.time()
            self.recording_paused_time = 0  # For pause/resume tracking
            self.current_frame_number = 0  # Track frame numbers for shot marking
        else:
            # Resuming after pause
            self.recording_paused_time += (time.time() - self.recording_pause_start)
        
        # Update UI to show recording is active
        self.recording_indicator.setVisible(True)
        
        # Start recording timer
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording)
        self.recording_timer.start(33)  # ~30 FPS
        
        # Set recording state
        self.is_recording = True

    def pause_recording(self):
        """Pause recording without finalizing."""
        if not hasattr(self, 'is_recording') or not self.is_recording:
            return
        
        # Stop timer
        if hasattr(self, 'recording_timer'):
            self.recording_timer.stop()
        
        # Mark pause time for later resuming
        self.recording_pause_start = time.time()
        
        # Hide the recording indicator while paused
        self.recording_indicator.setVisible(False)
        
        # Update state
        self.is_recording = False

    def update_recording(self):
        """Update recording with new frame and metrics data."""
        if not hasattr(self, 'is_recording') or not self.is_recording or not hasattr(self, 'video_writer'):
            return
        
        try:
            # Get the current frame with overlays
            frame = self.get_current_frame_with_overlays()
            
            if frame is not None:
                # Write frame
                self.video_writer.write(frame)
                
                # Update duration
                elapsed = time.time() - self.recording_start_time - self.recording_paused_time
                self.recording_metadata['duration'] = elapsed
                
                # Update recording indicator
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                self.recording_indicator.setText(f"● REC {minutes:02d}:{seconds:02d}")
                
                # Capture current metrics data - NEW CODE
                if hasattr(self, 'joint_tracker') and self.joint_tracker:
                    joint_history = self.joint_tracker.get_joint_history()
                    if joint_history:
                        current_metrics = {}
                        
                        # Calculate stability metrics
                        current_metrics['timestamp'] = elapsed  # Store time relative to recording start
                        
                        # Get latest joint positions
                        if joint_history and 'joints' in joint_history[-1]:
                            current_metrics['joint_positions'] = joint_history[-1]['joints']
                        
                        # Calculate sway velocity
                        if hasattr(self, 'stability_metrics'):
                            current_metrics['sway_velocity'] = self.stability_metrics.calculate_sway_velocity(joint_history)
                            dev_x, dev_y = self.stability_metrics.calculate_postural_stability(joint_history)
                            current_metrics['dev_x'] = dev_x
                            current_metrics['dev_y'] = dev_y
                            current_metrics['follow_through_score'] = self.stability_metrics.calculate_follow_through_score(
                                joint_history, 
                                shot_time=None,  # Not a shot frame
                                post_window=1.0
                            )
                        
                        # Add to metrics array
                        self.recording_metadata['metrics'].append(current_metrics)
                        
                        # Limit metrics array size to prevent huge files
                        # Keep approximately 5 frames per second for a minute-long recording
                        max_metrics = 5 * 60  # 5 fps × 60 seconds
                        if len(self.recording_metadata['metrics']) > max_metrics:
                            # Keep first few and most recent entries
                            keep_count = max_metrics // 2
                            self.recording_metadata['metrics'] = \
                                self.recording_metadata['metrics'][:keep_count] + \
                                self.recording_metadata['metrics'][-keep_count:]
        except Exception as e:
            print(f"Error updating recording: {e}")

    def get_current_frame_with_overlays(self):
        """Get the current camera frame with all visualization overlays."""
        # This is a simplified version - you'd need to adjust based on your actual UI layout
        if not hasattr(self.joint_tracker, 'cap') or not self.joint_tracker.cap.isOpened():
            return None
        
        # Get frame with joint tracking
        frame, joint_data, timestamp = self.joint_tracker.get_frame()
        
        if frame is None:
            return None
        
        # Add overlays similar to what's shown in the UI
        if hasattr(self, 'joint_tracker') and joint_data:
            # Get joint history for metrics
            joint_history = self.joint_tracker.get_joint_history()
            
            if joint_history:
                # Calculate stability metrics
                metrics = {}
                try:
                    metrics['sway_velocity'] = self.stability_metrics.calculate_sway_velocity(joint_history)
                    metrics['dev_x'], metrics['dev_y'] = self.stability_metrics.calculate_postural_stability(joint_history)
                    metrics['follow_through_score'] = self.stability_metrics.calculate_follow_through_score(
                        joint_history, time.time())
                    
                    # Draw stability heatmap
                    frame = self.draw_stability_heatmap(frame, metrics)
                    
                    # Add metrics text overlay
                    frame = self.add_metrics_overlay(frame, metrics)
                except Exception as e:
                    print(f"Error adding overlays: {str(e)}")
        
        return frame

    def save_current_metrics(self):
        """Save the current metrics to the recording metadata."""
        if not hasattr(self, 'recording_metadata'):
            return
        
        joint_history = self.joint_tracker.get_joint_history()
        if not joint_history:
            return
        
        # Calculate metrics
        try:
            metrics = {
                'timestamp': time.time() - self.recording_start_time,
                'sway_velocity': self.stability_metrics.calculate_sway_velocity(joint_history),
                'dev_x': self.stability_metrics.calculate_postural_stability(joint_history)[0],
                'dev_y': self.stability_metrics.calculate_postural_stability(joint_history)[1],
                'follow_through_score': self.stability_metrics.calculate_follow_through_score(
                    joint_history, time.time())
            }
            
            # Add to metadata
            self.recording_metadata['metrics'].append(metrics)
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")

    def stop_recording(self):
        """Stop and save the recording."""
        if not hasattr(self, 'is_recording'):
            return
        
        # Stop timer
        if hasattr(self, 'recording_timer'):
            self.recording_timer.stop()
        
        # Release video writer
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            self.video_writer = None
        
        # Save metadata
        if hasattr(self, 'recording_metadata') and hasattr(self, 'user_recordings_dir'):
            metadata_filename = os.path.splitext(self.video_filename)[0] + ".json"
            metadata_path = os.path.join(self.user_recordings_dir, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(self.recording_metadata, f, indent=4)
        
        # Hide recording indicator
        self.recording_indicator.setVisible(False)
        
        # Clear recording state
        self.is_recording = False
        delattr(self, 'recording_metadata')
        delattr(self, 'video_filename')
        delattr(self, 'video_path')
        
        # Show confirmation if appropriate
        if not hasattr(self, 'ending_session') or not self.ending_session:
            self.statusBar().showMessage("Recording saved successfully")

    # Helper methods for visualization overlays
    def draw_stability_heatmap(self, frame, metrics):
        """Draw stability heatmap overlay based on metrics."""
        # Create a copy to avoid modifying the original
        overlay = frame.copy()
        
        # Get joint positions from the latest frame
        joint_positions = {}
        for joint_name in self.joint_tracker.TRACKED_JOINTS:
            if joint_name in self.joint_tracker.joint_data.get('joints', {}):
                joint = self.joint_tracker.joint_data['joints'][joint_name]
                joint_positions[joint_name] = (int(joint['x']), int(joint['y']))
        
        # Get sway velocity for coloring
        sway_velocities = metrics.get('sway_velocity', {})
        
        # Draw heatmap circles for each joint
        for joint_name, (x, y) in joint_positions.items():
            sway = sway_velocities.get(joint_name, 0)
            
            # Size based on importance
            size = 30
            if 'WRIST' in joint_name or 'ELBOW' in joint_name:
                size = 40
            
            # Color based on stability
            if sway < 5.0:  # Stable - green
                color = (0, 255, 0)
            elif sway < 10.0:  # Medium - yellow/orange
                g = int(255 * (10.0 - sway) / 5.0)
                color = (0, g, 255)
            else:  # Unstable - red
                color = (0, 0, 255)
            
            # Draw circle
            cv2.circle(overlay, (x, y), size, color, -1)
        
        # Apply with transparency
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def add_metrics_overlay(self, frame, metrics):
        """Add text overlay with metrics information."""
        # Create background for text
        h, w = frame.shape[:2]
        x_offset = w - 260
        y_offset = 10
        overlay = np.zeros((150, 250, 3), dtype=np.uint8)
        overlay[:, :] = (40, 40, 40)  # Dark gray
        
        # Add metrics text
        stability_score = self._calculate_overall_stability(metrics)
        cv2.putText(overlay, f"Stability: {int(stability_score*100)}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        follow_through = metrics.get('follow_through_score', 0)
        cv2.putText(overlay, f"Follow-through: {follow_through:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate average sway
        sway_values = [metrics.get('sway_velocity', {}).get(joint, 0) for joint in 
                    ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW']]
        avg_sway = sum(sway_values) / max(1, len(sway_values))
        cv2.putText(overlay, f"Avg Sway: {avg_sway:.2f} mm/s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend overlay with frame
        roi = frame[y_offset:y_offset+150, x_offset:x_offset+250]
        overlay_alpha = 0.7
        cv2.addWeighted(overlay, overlay_alpha, roi, 1-overlay_alpha, 0, roi)
        frame[y_offset:y_offset+150, x_offset:x_offset+250] = roi
        
        return frame
    
    def toggle_recording_state(self):
        """Toggle the recording state without starting/stopping recording directly."""
        # Just update the button state - actual recording will start/stop with analysis
        is_enabled = self.record_enabled.isChecked()
        
        # Update feedback label instead of using statusBar
        current_feedback = self.feedback_label.text()
        if is_enabled:
            self.feedback_label.setText("Recording enabled - will start with analysis")
        else:
            self.feedback_label.setText("Recording disabled")
        
        # Restore original feedback after 2 seconds
        QTimer.singleShot(2000, lambda: self.feedback_label.setText(current_feedback))
        
        # If analysis is currently running and recording was just enabled, start recording now
        if is_enabled and self.camera_running and not hasattr(self, 'is_recording'):
            self.start_recording()
        
        # If analysis is running and recording was just disabled, stop recording now
        if not is_enabled and hasattr(self, 'is_recording') and self.is_recording:
            self.stop_recording()

    def start_analysis(self):
        """Start real-time analysis with improved error handling."""
        try:
            # Double-check session before starting
            if not hasattr(self, 'session_id') or not self.session_id:
                QMessageBox.warning(self, "No Active Session", 
                                "Please create a session first before starting analysis.")
                return
                
            # Reset shot detection and follow-through state
            if hasattr(self, 'last_shot_time'):
                delattr(self, 'last_shot_time')
                
            # Initialize components with error handling
            try:
                self.joint_tracker.start()
            except Exception as e:
                print(f"Error starting joint tracker: {e}")
                QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")
                return
            
            # Start audio detection
            try:
                self.start_audio_detection()
            except Exception as e:
                print(f"Error starting audio detection: {e}")
                # Continue even if audio fails
            
            # Start update timer
            self.update_timer.start(33)  # ~30 FPS
            
            # Update UI
            self.start_stop_button.setText("Stop Analysis")
            self.camera_running = True
            self.session_active = True
            
            if self.session_id:
                self.manual_shot_button.setEnabled(True)
            
            # Start recording if enabled
            if hasattr(self, 'record_enabled') and self.record_enabled.isChecked():
                try:
                    self.start_recording()
                except Exception as e:
                    print(f"Error starting recording: {e}")
                    # Continue even if recording fails
            
            # Update main window button if accessible
            try:
                main_window = self.window()
                if hasattr(main_window, 'session_button'):
                    main_window.session_button.setText("End Session")
            except Exception as e:
                print(f"Error updating main window: {e}")
                
        except Exception as e:
            print(f"Error in start_analysis: {e}")
            import traceback
            traceback.print_exc()
            self.camera_running = False
            QMessageBox.critical(self, "Error", f"Failed to start analysis: {str(e)}")

    def stop_analysis(self):
        """Stop real-time analysis and pause recording if active."""
        # Stop update timer
        self.update_timer.stop()
        
        # Stop joint tracker
        self.joint_tracker.stop()
        
        # Stop audio detection
        self.stop_audio_detection()
        
        # Pause recording if active
        if hasattr(self, 'is_recording') and self.is_recording:
            self.pause_recording()
        
        # Update UI
        self.start_stop_button.setText("Start Analysis")
        self.camera_running = False
        self.manual_shot_button.setEnabled(False)

    def complete_shot_processing(self):
        """Complete shot processing after collecting post-shot frames for follow-through analysis."""
        # Restore button state
        self.manual_shot_button.setEnabled(True)
        self.manual_shot_button.setText("Record Shot")
        
        # Retrieve the pending shot data
        if not hasattr(self, 'pending_shot_data'):
            print("Error: No pending shot data found")
            return
        
        # Get stored data
        shot_timestamp = self.pending_shot_data['timestamp']
        sway_velocities = self.pending_shot_data['sway_velocities']
        dev_x = self.pending_shot_data['dev_x']
        dev_y = self.pending_shot_data['dev_y']
        initial_positions = self.pending_shot_data['joint_positions']
        
        # Get the updated joint history which should now include post-shot frames
        updated_joint_history = self.joint_tracker.get_joint_history()
        
        initial_history_length = len(self.pending_shot_data['initial_joint_history'])
        updated_history_length = len(updated_joint_history)
        print(f"Original history length: {initial_history_length}")
        print(f"Updated history length: {updated_history_length}")
        print(f"New frames collected: {updated_history_length - initial_history_length}")
        
        # Debug timestamps
        if updated_history_length > 0:
            first_ts = updated_joint_history[0].get('timestamp', 0)
            last_ts = updated_joint_history[-1].get('timestamp', 0)
            print(f"Updated history range: {first_ts:.3f} to {last_ts:.3f} (span: {last_ts - first_ts:.3f}s)")
            print(f"Post-shot time available: {last_ts - shot_timestamp:.3f}s")
        
        # Calculate follow-through using the updated joint history
        print(f"Calculating follow-through with shot_time={shot_timestamp}")
        follow_through = self.stability_metrics.calculate_follow_through_score(
            updated_joint_history,
            shot_time=shot_timestamp,
            post_window=1.0
        )
        print(f"Follow-through score: {follow_through:.3f}")
        
        # Combine metrics
        metrics = {
            'sway_velocity': sway_velocities,
            'dev_x': dev_x,
            'dev_y': dev_y,
            'follow_through_score': follow_through,
            'joint_positions': initial_positions,
            'shot_time': shot_timestamp
        }
        
        # Use a custom dialog for decimal score entry
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox
        
        score_dialog = QDialog(self)
        score_dialog.setWindowTitle("Shot Recorded")
        score_dialog.setFixedWidth(300)
        
        layout = QVBoxLayout()
        
        # Add label with follow-through score for reference
        layout.addWidget(QLabel(f"Follow-through Score: {follow_through:.2f}"))
        layout.addWidget(QLabel("Enter score (0.00-10.9):"))
        
        # Create double spin box for decimal scores
        score_spinner = QDoubleSpinBox()
        score_spinner.setRange(0.0, 10.9)
        score_spinner.setDecimals(1)  # Allow one decimal place
        score_spinner.setSingleStep(0.1)
        score_spinner.setValue(10.9)  # Default to 10.9
        layout.addWidget(score_spinner)
        
        # Add buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(score_dialog.reject)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(score_dialog.accept)
        ok_button.setDefault(True)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)
        
        score_dialog.setLayout(layout)
        
        if score_dialog.exec() == QDialog.DialogCode.Accepted:
            score = score_spinner.value()
            
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
                    self.stability_metrics.update_baseline(updated_joint_history, score, current_best_score)
                    self.data_storage.update_baseline(
                        self.user_id, 
                        self.stability_metrics.baseline_metrics, 
                        score
                    )
                    QMessageBox.information(self, "New Baseline", 
                                        "New baseline set with this shot!")
                
                # Show confirmation
                QMessageBox.information(self, "Shot Recorded", 
                                    f"Shot recorded with score: {score}\nFollow-through: {follow_through:.2f}")
            else:
                QMessageBox.critical(self, "Error", "Failed to store shot data.")
        
        # Clear the pending shot data
        if hasattr(self, 'pending_shot_data'):
            delattr(self, 'pending_shot_data')
    
    def _capture_current_metrics(self):
        """Capture current frame metrics for recording with improved follow-through handling."""
        if not hasattr(self, 'joint_tracker') or not self.joint_tracker:
            return None
            
        joint_history = self.joint_tracker.get_joint_history()
        if not joint_history:
            return None
            
        current_metrics = {}
        
        # Calculate elapsed time
        elapsed = time.time() - self.recording_start_time - self.recording_paused_time
        current_metrics['timestamp'] = elapsed
        
        # Store current frame number if tracking it
        if hasattr(self, 'current_frame_number'):
            current_metrics['frame_number'] = self.current_frame_number
        
        # Get latest joint positions
        if joint_history and 'joints' in joint_history[-1]:
            current_metrics['joint_positions'] = joint_history[-1]['joints']
        
        # Calculate stability metrics
        if hasattr(self, 'stability_metrics'):
            try:
                # Always calculate sway and deviation
                current_metrics['sway_velocity'] = self.stability_metrics.calculate_sway_velocity(joint_history)
                dev_x, dev_y = self.stability_metrics.calculate_postural_stability(joint_history)
                current_metrics['dev_x'] = dev_x
                current_metrics['dev_y'] = dev_y
                
                # Only calculate follow-through if we have a shot time
                if hasattr(self, 'last_shot_time') and self.last_shot_time:
                    time_since_shot = time.time() - self.last_shot_time
                    # Only calculate follow-through if within 3 seconds after shot
                    if time_since_shot < 3.0 and len(joint_history) >= 5:
                        current_metrics['follow_through_score'] = self.stability_metrics.calculate_follow_through_score(
                            joint_history, 
                            shot_time=self.last_shot_time,
                            post_window=1.0
                        )
                    else:
                        current_metrics['follow_through_score'] = 0.0
                else:
                    # No shot detected yet, use a default value instead of calculating
                    current_metrics['follow_through_score'] = 0.0
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                # Add basic empty structures to avoid errors during playback
                current_metrics['sway_velocity'] = {}
                current_metrics['dev_x'] = {}
                current_metrics['dev_y'] = {}
                current_metrics['follow_through_score'] = 0.0
        
        return current_metrics