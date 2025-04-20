from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QGridLayout, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import numpy as np
import cv2
from typing import Dict, List, Optional
import time
import json
import os

from src.data_storage import DataStorage

class ReplayWidget(QWidget):
    """
    Widget for replaying and analyzing recorded shooting sessions.
    Allows playback of recordings with analysis overlays.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the replay widget.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        self.session_id = None
        
        # Replay state
        self.is_playing = False
        self.current_recording = None
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback)
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface elements."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top control bar
        control_layout = QHBoxLayout()
        
        # Session selector
        control_layout.addWidget(QLabel("Session:"))
        self.session_selector = QComboBox()
        self.session_selector.setMinimumWidth(200)
        self.session_selector.currentIndexChanged.connect(self.on_session_changed)
        control_layout.addWidget(self.session_selector)
        
        # Add more spacing
        control_layout.addStretch()
        
        # Add playback recordings label (bold)
        sessions_label = QLabel("Playback Recordings:")
        sessions_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(sessions_label)
        
        # Status label
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)
        
        main_layout.addLayout(control_layout)
        
        # Main content splitter (recordings list and playback)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Recordings list
        recordings_widget = QWidget()
        recordings_layout = QVBoxLayout()
        recordings_layout.addWidget(QLabel("Recorded Sessions:"))
        
        self.recordings_list = QListWidget()
        self.recordings_list.itemSelectionChanged.connect(self.on_recording_selected)
        recordings_layout.addWidget(self.recordings_list)
        
        # Add refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload))
        self.refresh_button.clicked.connect(self.load_recordings)
        self.refresh_button.setToolTip("Refresh the recordings list")
        recordings_layout.addWidget(self.refresh_button)
        
        # Delete recording button
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_recording)
        recordings_layout.addWidget(self.delete_button)
        
        recordings_widget.setLayout(recordings_layout)
        main_splitter.addWidget(recordings_widget)
        
        # Right side: Playback
        playback_widget = QWidget()
        playback_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("No video selected")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        playback_layout.addWidget(self.video_label)
        
        # Playback controls
        playback_controls = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        self.play_button.setToolTip("Play selected recording")
        playback_controls.addWidget(self.play_button)
        
        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self.seek_playback)
        playback_controls.addWidget(self.timeline_slider)
        
        # Time label
        self.time_label = QLabel("0:00 / 0:00")
        playback_controls.addWidget(self.time_label)
        
        playback_layout.addLayout(playback_controls)
        
        # Playback options
        options_group = QGroupBox("Playback Options")
        options_layout = QVBoxLayout()
        
        # Overlay checkboxes
        overlay_layout = QHBoxLayout()
        
        self.skeleton_checkbox = QCheckBox("Skeleton Overlay")
        self.skeleton_checkbox.setChecked(True)
        self.skeleton_checkbox.stateChanged.connect(self.update_playback_options)
        overlay_layout.addWidget(self.skeleton_checkbox)
        
        self.heatmap_checkbox = QCheckBox("Stability Heatmap")
        self.heatmap_checkbox.setChecked(True)
        self.heatmap_checkbox.stateChanged.connect(self.update_playback_options)
        overlay_layout.addWidget(self.heatmap_checkbox)
        
        self.metrics_checkbox = QCheckBox("Show Metrics")
        self.metrics_checkbox.setChecked(True)
        self.metrics_checkbox.stateChanged.connect(self.update_playback_options)
        overlay_layout.addWidget(self.metrics_checkbox)
        
        options_layout.addLayout(overlay_layout)
        
        # Playback speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Playback Speed:"))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # 1.0x is default
        self.speed_combo.currentIndexChanged.connect(self.update_playback_speed)
        speed_layout.addWidget(self.speed_combo)
        
        options_layout.addLayout(speed_layout)
        
        options_group.setLayout(options_layout)
        playback_layout.addWidget(options_group)
        
        # Session metrics
        metrics_group = QGroupBox("Session Metrics")
        metrics_layout = QVBoxLayout()
        
        self.metrics_label = QLabel("Select a recording to view metrics.")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        
        # Stability gauge
        self.stability_gauge = QProgressBar()
        self.stability_gauge.setMinimum(0)
        self.stability_gauge.setMaximum(100)
        self.stability_gauge.setValue(0)
        self.stability_gauge.setTextVisible(True)
        self.stability_gauge.setFormat("Stability: %v%")
        metrics_layout.addWidget(self.stability_gauge)
        
        metrics_group.setLayout(metrics_layout)
        playback_layout.addWidget(metrics_group)
        
        playback_widget.setLayout(playback_layout)
        main_splitter.addWidget(playback_widget)
        
        # Set initial sizes
        main_splitter.setSizes([200, 800])
        
        main_layout.addWidget(main_splitter)
        
        self.setLayout(main_layout)
        
        # Initialize recordings directory
        self.recordings_dir = "data/recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
    
    def set_user(self, user_id: int):
        """
        Set the current user and load their sessions.
        
        Args:
            user_id: ID of the current user
        """
        self.user_id = user_id
        self.refresh_sessions()
        
        # Create user-specific recordings directory
        self.user_recordings_dir = os.path.join(self.recordings_dir, f"user_{user_id}")
        os.makedirs(self.user_recordings_dir, exist_ok=True)
        
        # Load recordings
        self.load_recordings()
    
    def refresh_sessions(self):
        """Refresh the sessions dropdown with user's sessions."""
        if not self.user_id:
            return
        
        # Clear current items
        self.session_selector.clear()
        
        # Add placeholder
        self.session_selector.addItem("Select a session...", -1)
        
        # Get sessions from database
        sessions = self.data_storage.get_sessions(self.user_id)
        
        # Add sessions to dropdown
        for session in sessions:
            session_text = f"{session['name']} ({session['created_at'][:10]})"
            self.session_selector.addItem(session_text, session['id'])
    
    def set_session(self, session_id: int):
        """
        Set the current session.
        
        Args:
            session_id: ID of the session
        """
        # Update selector to match
        for i in range(self.session_selector.count()):
            if self.session_selector.itemData(i) == session_id:
                self.session_selector.setCurrentIndex(i)
                return
        
        # If not found in selector, set manually
        self.session_id = session_id
    
    def on_session_changed(self, index: int):
        """
        Handle session selection change.
        
        Args:
            index: Index of the selected session in the dropdown
        """
        if index <= 0:  # "Select a session..." item
            self.session_id = None
            return
        
        # Get session ID from combobox data
        session_id = self.session_selector.itemData(index)
        
        if session_id > 0:
            self.session_id = session_id
    
    def load_recordings(self):
        """Load recordings for the current user."""
        if not self.user_id:
            return
        
        # Clear list
        self.recordings_list.clear()
        
        # Check if directory exists
        user_recordings_dir = os.path.join("data/recordings", f"user_{self.user_id}")
        if not os.path.exists(user_recordings_dir):
            return
        
        # Get all recording metadata files
        for filename in os.listdir(user_recordings_dir):
            if filename.endswith(".json"):
                # Load metadata
                metadata_path = os.path.join(user_recordings_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if video file exists
                    video_path = os.path.join(user_recordings_dir, metadata.get('video_file', ''))
                    if not os.path.exists(video_path):
                        continue
                    
                    # Create list item
                    session_name = metadata.get('session_name', 'Unknown')
                    timestamp = metadata.get('timestamp', 'Unknown')
                    duration = metadata.get('duration', 0)
                    
                    # Format duration as mm:ss
                    duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
                    
                    item_text = f"{session_name} - {timestamp} ({duration_str})"
                    item = QListWidgetItem(item_text)
                    
                    # Store metadata as item data
                    item.setData(Qt.ItemDataRole.UserRole, metadata)
                    
                    self.recordings_list.addItem(item)
                    
                except Exception as e:
                    print(f"Error loading recording metadata: {str(e)}")
    
    def on_recording_selected(self):
        """Handle recording selection change."""
        selected_items = self.recordings_list.selectedItems()
        
        if not selected_items:
            self.play_button.setEnabled(False)
            self.current_recording = None
            self.video_label.setText("No video selected")
            self.time_label.setText("0:00 / 0:00")
            self.timeline_slider.setValue(0)
            self.metrics_label.setText("Select a recording to view metrics.")
            return
        
        # Get selected recording metadata
        item = selected_items[0]
        metadata = item.data(Qt.ItemDataRole.UserRole)
        self.current_recording = metadata
        
        # Open video file
        video_path = os.path.join(self.user_recordings_dir, metadata.get('video_file', ''))
        if not os.path.exists(video_path):
            self.play_button.setEnabled(False)
            self.video_label.setText("Video file not found")
            return
        
        # Initialize video capture
        self.playback_cap = cv2.VideoCapture(video_path)
        if not self.playback_cap.isOpened():
            self.play_button.setEnabled(False)
            self.video_label.setText("Could not open video")
            return
        
        # Get video properties
        self.playback_fps = self.playback_cap.get(cv2.CAP_PROP_FPS)
        self.playback_frame_count = int(self.playback_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.playback_duration = self.playback_frame_count / self.playback_fps
        
        # Update timeline slider
        self.timeline_slider.setMaximum(self.playback_frame_count - 1)
        self.timeline_slider.setValue(0)
        
        # Update time label
        minutes = int(self.playback_duration // 60)
        seconds = int(self.playback_duration % 60)
        self.time_label.setText(f"0:00 / {minutes}:{seconds:02d}")
        
        # Display first frame
        ret, frame = self.playback_cap.read()
        if ret:
            self.update_display(frame)
        
        # Update metrics
        self.update_metrics_display(metadata)
        
        # Enable play button
        self.play_button.setEnabled(True)
    
    def toggle_playback(self):
        """Toggle playback state."""
        if not self.current_recording:
            return
        
        if not self.is_playing:
            self.start_playback()
        else:
            self.pause_playback()
    
    def start_playback(self):
        """Start video playback."""
        if not self.current_recording or not self.playback_cap.isOpened():
            return
        
        # Get current position
        current_frame = self.timeline_slider.value()
        self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Update UI
        self.play_button.setText("Pause")
        self.is_playing = True
        
        # Start timer at appropriate interval based on FPS
        playback_speed = self.get_playback_speed()
        interval = int(1000 / (self.playback_fps * playback_speed))
        self.playback_timer.start(interval)
    
    def pause_playback(self):
        """Pause video playback."""
        if not self.is_playing:
            return
        
        # Stop timer
        self.playback_timer.stop()
        
        # Update UI
        self.play_button.setText("Play")
        self.is_playing = False
    
    def update_playback(self):
        """Update video playback with next frame."""
        if not self.is_playing or not self.playback_cap.isOpened():
            return
        
        # Read next frame
        ret, frame = self.playback_cap.read()
        if not ret:
            # End of video
            self.pause_playback()
            self.timeline_slider.setValue(0)
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        # Update display
        self.update_display(frame)
        
        # Update position
        current_frame = int(self.playback_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.timeline_slider.setValue(current_frame)
        
        # Update time label
        current_time = current_frame / self.playback_fps
        total_time = self.playback_duration
        
        current_minutes = int(current_time // 60)
        current_seconds = int(current_time % 60)
        
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        
        self.time_label.setText(f"{current_minutes}:{current_seconds:02d} / "
                               f"{total_minutes}:{total_seconds:02d}")
    
    def seek_playback(self, position: int):
        """
        Seek to a specific position in the video.
        
        Args:
            position: Frame number to seek to
        """
        if not self.current_recording or not self.playback_cap.isOpened():
            return
        
        # Set position
        self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        # Read frame at new position
        ret, frame = self.playback_cap.read()
        if ret:
            self.update_display(frame)
            
            # Set position back by 1 so next read gets the next frame
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        # Update time label
        current_time = position / self.playback_fps
        total_time = self.playback_duration
        
        current_minutes = int(current_time // 60)
        current_seconds = int(current_time % 60)
        
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        
        self.time_label.setText(f"{current_minutes}:{current_seconds:02d} / "
                               f"{total_minutes}:{total_seconds:02d}")
    
    def update_display(self, frame: np.ndarray):
        """
        Update the video display with the current frame.
        
        Args:
            frame: OpenCV image array
        """
        if frame is None:
            return
        
        # Add overlays if needed
        frame = self.add_overlays(frame)
        
        # Convert frame to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale image to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), 
                                                Qt.AspectRatioMode.KeepAspectRatio))
    
    def add_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Add overlays to the frame based on playback options.
        
        Args:
            frame: OpenCV image array
                
        Returns:
            Frame with overlays
        """
        # Check if we have a recording with metrics
        if not self.current_recording or not self.is_playing:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        overlay_frame = frame.copy()
        
        # Calculate current frame position as proportion of total
        frame_position = self.timeline_slider.value() / max(1, self.timeline_slider.maximum())
        
        # Add skeleton overlay with proper joint positions
        if self.skeleton_checkbox.isChecked():
            # Generate simulated joint positions based on frame position
            # In a real app, these would come from the actual recording data
            joint_positions = self._generate_joint_positions(frame_position)
            
            # Draw the skeleton with connections
            if joint_positions:
                self._draw_skeleton(overlay_frame, joint_positions)
        
        # Add heatmap overlay based on joint stability
        if self.heatmap_checkbox.isChecked():
            # Generate stability metrics based on frame position
            # In a real app, these would come from the actual recording data
            stability_metrics = self._generate_stability_metrics(frame_position)
            
            # Draw the heatmap
            if stability_metrics:
                overlay_frame = self._draw_stability_heatmap(overlay_frame, stability_metrics)
        
        # Add metrics overlay
        if self.metrics_checkbox.isChecked():
            # Add text with metrics
            overlay_frame = self._add_metrics_overlay(overlay_frame, frame_position)
        
        return overlay_frame
    
    def update_playback_options(self):
        """Update playback options based on checkbox states."""
        # If we're playing, force an update to the display
        if self.is_playing and self.playback_cap.isOpened():
            # Get current position
            current_frame = int(self.playback_cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Read current frame again
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
            ret, frame = self.playback_cap.read()
            
            if ret:
                self.update_display(frame)
    
    def update_playback_speed(self):
        """Update playback speed based on combobox selection."""
        if not self.is_playing:
            return
        
        # Get speed
        playback_speed = self.get_playback_speed()
        
        # Update timer interval
        interval = int(1000 / (self.playback_fps * playback_speed))
        self.playback_timer.start(interval)
    
    def get_playback_speed(self) -> float:
        """
        Get the current playback speed multiplier.
        
        Returns:
            Playback speed as float
        """
        speed_text = self.speed_combo.currentText()
        return float(speed_text.rstrip('x'))
    
    def update_metrics_display(self, metadata: Dict):
        """
        Update the metrics display with recording metrics.
        
        Args:
            metadata: Recording metadata dictionary
        """
        # This is a placeholder implementation since we don't have actual
        # metrics in the recording metadata yet
        
        session_name = metadata.get('session_name', 'Unknown')
        timestamp = metadata.get('timestamp', 'Unknown')
        duration = metadata.get('duration', 0)
        
        info_text = f"Session: {session_name}\n"
        info_text += f"Recorded: {timestamp}\n"
        info_text += f"Duration: {int(duration // 60)}:{int(duration % 60):02d}\n\n"
        
        # Add placeholder metrics
        info_text += "Average Stability Score: 75%\n"
        info_text += "Best Shot: 8.5/10\n"
        info_text += "Average Follow-through: 0.72\n"
        
        self.metrics_label.setText(info_text)
        
        # Update stability gauge
        self.stability_gauge.setValue(75)
    
    def delete_recording(self):
        """Delete the selected recording."""
        selected_items = self.recordings_list.selectedItems()
        
        if not selected_items:
            return
        
        # Get selected recording metadata
        item = selected_items[0]
        metadata = item.data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Delete Recording", 
            "Are you sure you want to delete this recording?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Stop playback if playing this recording
        if self.current_recording == metadata:
            self.pause_playback()
            if self.playback_cap and self.playback_cap.isOpened():
                self.playback_cap.release()
            
            self.current_recording = None
            self.video_label.setText("No video selected")
            self.time_label.setText("0:00 / 0:00")
            self.timeline_slider.setValue(0)
            self.metrics_label.setText("Select a recording to view metrics.")
            self.play_button.setEnabled(False)
        
        # Delete files
        video_path = os.path.join(self.user_recordings_dir, metadata.get('video_file', ''))
        metadata_path = os.path.join(
            self.user_recordings_dir, 
            os.path.splitext(metadata.get('video_file', ''))[0] + ".json"
        )
        
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Remove from list
            row = self.recordings_list.row(item)
            self.recordings_list.takeItem(row)
            
            self.status_label.setText("Recording deleted")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete recording: {str(e)}")
    
    def closeEvent(self, event):
        """Handle widget close event."""
        # Stop playback if active
        if self.is_playing:
            self.pause_playback()
        
        # Release resources
        if hasattr(self, 'playback_cap') and self.playback_cap and self.playback_cap.isOpened():
            self.playback_cap.release()
        
        event.accept()

    def set_session(self, session_id: int):
        """Set the current session with proper refresh logic."""
        if session_id <= 0:
            return
        
        # Force refresh recordings list after setting session
        self.session_id = session_id
        
        # Clear current selection
        self.current_recording = None
        
        # Update the session selector
        for i in range(self.session_selector.count()):
            if self.session_selector.itemData(i) == session_id:
                self.session_selector.setCurrentIndex(i)
                break
        
        # Reload recordings list
        self.load_recordings()

    def _generate_joint_positions(self, frame_position):
        """
        Generate simulated joint positions based on frame position.
        In a real app, this would retrieve actual recording data.
        
        Args:
            frame_position: Position in the playback (0-1)
                
        Returns:
            Dictionary of joint positions
        """
        # Base positions for common joints in pixel coordinates
        base_positions = {
            'NOSE': (320, 100),
            'LEFT_SHOULDER': (280, 150),
            'RIGHT_SHOULDER': (360, 150),
            'LEFT_ELBOW': (250, 200),
            'RIGHT_ELBOW': (390, 200),
            'LEFT_WRIST': (220, 250),
            'RIGHT_WRIST': (420, 250),
            'LEFT_HIP': (290, 300),
            'RIGHT_HIP': (350, 300),
            'LEFT_KNEE': (285, 380),
            'RIGHT_KNEE': (355, 380),
            'LEFT_ANKLE': (280, 450),
            'RIGHT_ANKLE': (360, 450)
        }
        
        # Add some movement based on frame position
        # We'll use sine waves to create natural-looking motion
        import math
        
        # Create amplitude that increases toward middle of playback and decreases toward end
        # This simulates a shot where motion peaks at trigger pull (middle of recording)
        motion_amplitude = 10 * math.sin(frame_position * math.pi)
        
        # Different joints have different motion patterns
        joint_positions = {}
        
        for joint, (x, y) in base_positions.items():
            # Different offsets for each joint
            x_offset = motion_amplitude * math.sin(frame_position * 2 * math.pi + hash(joint) % 10)
            y_offset = motion_amplitude * math.cos(frame_position * 2 * math.pi + hash(joint) % 10)
            
            # Wrists and elbows move more
            if 'WRIST' in joint or 'ELBOW' in joint:
                x_offset *= 1.5
                y_offset *= 1.5
            
            # Nose has specific pattern
            if joint == 'NOSE':
                y_offset *= 0.5  # Less vertical movement
            
            # Calculate final position
            joint_positions[joint] = (int(x + x_offset), int(y + y_offset))
        
        return joint_positions

    def _generate_stability_metrics(self, frame_position):
        """
        Generate simulated stability metrics based on frame position.
        In a real app, this would retrieve actual recording data.
        
        Args:
            frame_position: Position in the playback (0-1)
                
        Returns:
            Dictionary of stability metrics for joints
        """
        import math
        
        # Create stability metrics that deteriorate toward the middle (trigger pull)
        # and then improve again toward the end (follow through)
        
        # Stability is worst at trigger pull (middle of playback)
        base_stability = 1.0 - math.sin(frame_position * math.pi)
        
        # Different joints have different stability patterns
        joint_stability = {}
        
        # Common joints
        joints = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
            'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]
        
        for joint in joints:
            # Add some variation for each joint
            joint_factor = 0.8 + (hash(joint) % 10) / 20.0  # Range 0.8-1.3
            
            # Calculate stability (0-1, higher is more stable)
            stability = base_stability * joint_factor
            
            # Ensure within valid range
            stability = max(0.0, min(1.0, stability))
            
            # Wrists and elbows are typically less stable
            if 'WRIST' in joint:
                stability *= 0.7
            elif 'ELBOW' in joint:
                stability *= 0.85
            
            # Add to dictionary
            joint_stability[joint] = stability
        
        return joint_stability

    def _draw_skeleton(self, frame, joint_positions):
        """
        Draw skeleton overlay on the frame.
        
        Args:
            frame: OpenCV image
            joint_positions: Dictionary of joint positions
        """
        # Define connections for skeleton
        connections = [
            ('NOSE', 'LEFT_SHOULDER'),
            ('NOSE', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE')
        ]
        
        # Draw joints
        for joint, (x, y) in joint_positions.items():
        # Ensure coordinates are within frame dimensions
            x = max(0, min(x, frame.shape[1]-1))
            y = max(0, min(y, frame.shape[0]-1))
            
            # Draw larger, more visible circles
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), -1)  # Filled circle
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 0), 2)       # Black outline
        
        # Draw connections
        for joint1, joint2 in connections:
            if joint1 in joint_positions and joint2 in joint_positions:
                pt1 = joint_positions[joint1]
                pt2 = joint_positions[joint2]
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), (0, 255, 255), 2) # Yellow lines
        
        return frame

    def _draw_stability_heatmap(self, frame, stability_metrics):
        """
        Draw stability heatmap overlay on the frame.
        
        Args:
            frame: OpenCV image
            stability_metrics: Dictionary of stability values per joint
                
        Returns:
            Frame with heatmap overlay
        """
        # Create a transparent overlay
        overlay = frame.copy()
        
        # Get joint positions based on current frame
        frame_position = self.timeline_slider.value() / max(1, self.timeline_slider.maximum())
        joint_positions = self._generate_joint_positions(frame_position)
        
        # Draw heatmap circles for each joint based on stability
        for joint, (x, y) in joint_positions.items():
            if joint in stability_metrics:
                stability = stability_metrics[joint]
                
                # Size based on importance of joint
                size = 30
                if 'WRIST' in joint or 'ELBOW' in joint:
                    size = 40  # Larger for important shooting joints
                elif 'NOSE' in joint:
                    size = 35  # Important for sight alignment
                elif 'SHOULDER' in joint:
                    size = 35  # Important for shooting
                
                # Color based on stability (red=unstable, green=stable)
                if stability < 0.3:
                    color = (0, 0, 255)  # Red (unstable)
                elif stability < 0.7:
                    # Gradient from red to yellow to green
                    g = int(255 * (stability - 0.3) / 0.4)
                    color = (0, g, 255)  # Yellow-orange
                else:
                    color = (0, 255, 0)  # Green (stable)
                
                # Draw a filled circle with transparency
                cv2.circle(overlay, (x, y), size, color, -1)
                
                # Add stability value text
                cv2.putText(overlay, f"{stability:.2f}", (x-15, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Apply the overlay with transparency
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add legend
        legend_y = 30
        # Green - Stable
        cv2.circle(frame, (30, legend_y), 10, (0, 255, 0), -1)
        cv2.putText(frame, "Stable", (50, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Yellow - Medium
        cv2.circle(frame, (150, legend_y), 10, (0, 255, 255), -1)
        cv2.putText(frame, "Medium", (170, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Red - Unstable
        cv2.circle(frame, (270, legend_y), 10, (0, 0, 255), -1)
        cv2.putText(frame, "Unstable", (290, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def _add_metrics_overlay(self, frame, frame_position):
        """
        Add metrics text overlay to the frame.
        
        Args:
            frame: OpenCV image
            frame_position: Position in playback (0-1)
                
        Returns:
            Frame with metrics overlay
        """
        # Create metrics based on frame position
        import math
        
        # Simulate key metrics that change throughout the recording
        follow_through = 0.3 + 0.7 * (1 - math.sin(frame_position * math.pi))  # Best at end
        avg_sway = 15 * math.sin(frame_position * math.pi)  # Worst at middle
        stability_score = int(50 + 50 * (1 - math.sin(frame_position * math.pi)))  # Percentage
        
        # Create background box for metrics
        metrics_bg = np.zeros((150, 250, 3), dtype=np.uint8)
        metrics_bg[:, :] = (40, 40, 40)  # Dark gray background
        
        # Add metrics text
        cv2.putText(metrics_bg, f"Stability: {stability_score}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(metrics_bg, f"Follow-through: {follow_through:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(metrics_bg, f"Avg Sway: {avg_sway:.2f} mm/s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Position and blend the metrics box
        h, w = frame.shape[:2]
        x_offset = w - 260
        y_offset = 10
        
        # Create a region of interest
        roi = frame[y_offset:y_offset+150, x_offset:x_offset+250]
        
        # Blend the metrics box with the ROI
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(metrics_bg, alpha, roi, 1-alpha, 0, roi)
        
        # Put the ROI back into the frame
        frame[y_offset:y_offset+150, x_offset:x_offset+250] = roi
        
        return frame