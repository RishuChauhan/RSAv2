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
    Widget for recording and replaying shooting sessions with analysis overlays.
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
        self.is_recording = False
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
        
        # Recording controls
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)
        
        control_layout.addStretch()
        
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
            self.record_button.setEnabled(False)
            return
        
        # Get session ID from combobox data
        session_id = self.session_selector.itemData(index)
        
        if session_id > 0:
            self.session_id = session_id
            self.record_button.setEnabled(True)
    
    def load_recordings(self):
        """Load recordings for the current user."""
        if not self.user_id:
            return
        
        # Clear list
        self.recordings_list.clear()
        
        # Check if directory exists
        if not os.path.exists(self.user_recordings_dir):
            return
        
        # Get all recording metadata files
        for filename in os.listdir(self.user_recordings_dir):
            if filename.endswith(".json"):
                # Load metadata
                metadata_path = os.path.join(self.user_recordings_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if video file exists
                    video_path = os.path.join(self.user_recordings_dir, metadata.get('video_file', ''))
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
    
    def toggle_recording(self):
        """Toggle recording state."""
        if not self.session_id:
            return
        
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording the current session."""
        # Get session details
        self.cursor = self.data_storage.conn.cursor()
        self.cursor.execute("SELECT name FROM sessions WHERE id = ?", (self.session_id,))
        session = self.cursor.fetchone()
        
        if not session:
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", "Could not open camera.")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_filename = f"session_{self.session_id}_{timestamp}.mp4"
        video_path = os.path.join(self.user_recordings_dir, video_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Create metadata
        self.recording_metadata = {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'session_name': session['name'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'video_file': video_filename,
            'width': width,
            'height': height,
            'fps': fps,
            'metrics': [],
            'duration': 0
        }
        
        # Update UI
        self.record_button.setText("Stop Recording")
        self.status_label.setText("Recording...")
        self.is_recording = True
        
        # Start recording timer
        self.recording_start_time = time.time()
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording)
        self.recording_timer.start(33)  # ~30 FPS
    
    def update_recording(self):
        """Update recording with new frame."""
        if not self.is_recording or not self.cap.isOpened():
            return
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.stop_recording()
            return
        
        # Record frame
        self.video_writer.write(frame)
        
        # Display frame
        self.update_display(frame)
        
        # Update duration in metadata
        self.recording_metadata['duration'] = time.time() - self.recording_start_time
        
        # Update time label
        duration = self.recording_metadata['duration']
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        self.time_label.setText(f"{minutes}:{seconds:02d}")
    
    def stop_recording(self):
        """Stop recording and save metadata."""
        if not self.is_recording:
            return
        
        # Stop timer
        self.recording_timer.stop()
        
        # Release resources
        self.video_writer.release()
        self.cap.release()
        
        # Save metadata
        metadata_filename = os.path.splitext(self.recording_metadata['video_file'])[0] + ".json"
        metadata_path = os.path.join(self.user_recordings_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.recording_metadata, f, indent=4)
        
        # Update UI
        self.record_button.setText("Start Recording")
        self.status_label.setText("Recording saved")
        self.is_recording = False
        
        # Reload recordings list
        self.load_recordings()
    
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
        
        # This is a placeholder implementation, since we would need actual
        # joint tracking data from recording to add real overlays
        
        # Add skeleton overlay
        if self.skeleton_checkbox.isChecked():
            # Draw placeholder skeleton (just a rectangle)
            cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 2)
        
        # Add heatmap overlay
        if self.heatmap_checkbox.isChecked():
            # Draw placeholder heatmap (just a circle)
            cv2.circle(frame, (300, 300), 50, (0, 0, 255), -1)
            cv2.circle(frame, (300, 300), 50, (0, 0, 255, 128), -1)
        
        # Add metrics overlay
        if self.metrics_checkbox.isChecked():
            # Add text
            cv2.putText(frame, "Stability Score: 75%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
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
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Stop playback if active
        if self.is_playing:
            self.pause_playback()
        
        # Release resources
        if hasattr(self, 'playback_cap') and self.playback_cap and self.playback_cap.isOpened():
            self.playback_cap.release()
        
        event.accept()