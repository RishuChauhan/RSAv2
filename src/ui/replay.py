from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QGridLayout, QCheckBox, QProgressBar, QFrame,
    QDialog, QTextEdit, QDialogButtonBox, QFileDialog, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

from PyQt6.QtWidgets import QSizePolicy

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import time
import json
import os
import math
import datetime

from src.data_storage import DataStorage

class AnnotationLayer(QWidget):
    """Widget for adding annotations to video frames."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = []
        self.current_annotation = None
        self.annotation_color = QColor(255, 50, 50, 200)  # Semi-transparent red
        self.annotation_width = 3
        self.drawing = False
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def paintEvent(self, event):
        """Draw annotations on the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw existing annotations
        pen = QPen(self.annotation_color, self.annotation_width)
        painter.setPen(pen)
        
        for annotation in self.annotations:
            painter.drawLine(annotation[0], annotation[1])
        
        # Draw current annotation if being drawn
        if self.drawing and self.current_annotation:
            painter.drawLine(self.current_annotation[0], self.current_annotation[1])
    
    def mousePressEvent(self, event):
        """Handle mouse press events for annotation drawing."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.current_annotation = (QPointF(event.position()), QPointF(event.position()))
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for annotation drawing."""
        if self.drawing and self.current_annotation:
            self.current_annotation = (self.current_annotation[0], QPointF(event.position()))
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for annotation drawing."""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.current_annotation:
                # Only add annotation if it's not just a click (has some length)
                start = self.current_annotation[0]
                end = QPointF(event.position())
                
                # Calculate length of line
                length = math.sqrt((end.x() - start.x())**2 + (end.y() - start.y())**2)
                
                if length > 5:  # Only add if longer than 5 pixels
                    self.annotations.append((start, end))
                
                self.current_annotation = None
                self.update()
    
    def clear_annotations(self):
        """Clear all annotations."""
        self.annotations = []
        self.current_annotation = None
        self.update()
    
    def set_color(self, color):
        """Set annotation color."""
        self.annotation_color = color
    
    def set_width(self, width):
        """Set annotation width."""
        self.annotation_width = width

class VideoFrameWidget(QWidget):
    """Enhanced widget for displaying video frames with annotations."""
    
    def __init__(self):
        """Initialize the video frame widget with annotation capability."""
        super().__init__()
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Add frame container
        self.frame_container = QFrame()
        self.frame_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_container.setStyleSheet("""
            QFrame {
                border: 2px solid #1E88E5;
                border-radius: 4px;
                background-color: #263238;
            }
        """)
        
        # Container layout
        self.container_layout = QVBoxLayout(self.frame_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display label
        self.video_label = QLabel("No video selected")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("color: white; font-size: 16px;")
        self.container_layout.addWidget(self.video_label)
        
        # Annotation layer on top of video
        self.annotation_layer = AnnotationLayer(self.frame_container)
        self.annotation_layer.setGeometry(self.video_label.geometry())
        self.annotation_layer.lower()  # Ensure it's behind the video label
        
        # Add frame container to main layout
        self.layout.addWidget(self.frame_container)
        
        # Set up size policy
        self.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
        )
    
    def resizeEvent(self, event):
        """Handle resize events to ensure annotation layer matches video size."""
        super().resizeEvent(event)
        # Update annotation layer size to match video label
        if hasattr(self, 'annotation_layer') and hasattr(self, 'video_label'):
            self.annotation_layer.setGeometry(self.video_label.geometry())
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the displayed frame with enhanced error handling.
        
        Args:
            frame: OpenCV image array
        """
        try:
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
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio
            ))
            
            # Ensure annotation layer is visible and correctly sized
            self.annotation_layer.raise_()
            self.annotation_layer.setGeometry(self.video_label.geometry())
            
        except Exception as e:
            print(f"Error updating frame: {e}")
            self.video_label.setText(f"Error displaying frame: {str(e)}")
    
    def clear(self):
        """Clear the video display."""
        self.video_label.clear()
        self.video_label.setText("No video selected")
        self.annotation_layer.clear_annotations()

class ReplayWidget(QWidget):
    """
    Advanced widget for replaying and analyzing recorded shooting sessions.
    Designed specifically for rifle shooting coaches with enhanced analysis tools.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the enhanced replay widget with coaching features.
        
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
        self.frame_rate = 30  # Default frame rate
        self.current_frame = 0
        self.total_frames = 0
        
        # Initialize playback control timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback)
        
        # Shot detection flag
        self.shot_frame_marked = False
        self.shot_frame = 0
        
        # Coach annotations
        self.annotations = []
        self.notes = {}  # Frame-indexed notes
        
        # Initialize UI
        self.init_ui()
        
        # Initialize recording directory path
        self.recordings_dir = "data/recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
    
    def init_ui(self):
        """Initialize the user interface with enhanced coaching tools."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top bar with session and recording controls
        top_layout = QHBoxLayout()
        
        # Session selector
        top_layout.addWidget(QLabel("Session:"))
        self.session_selector = QComboBox()
        self.session_selector.setMinimumWidth(200)
        self.session_selector.currentIndexChanged.connect(self.on_session_changed)
        top_layout.addWidget(self.session_selector)
        
        # Add spacer
        top_layout.addStretch()
        
        # Create refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip("Refresh recordings list")
        self.refresh_button.clicked.connect(self.load_recordings)
        top_layout.addWidget(self.refresh_button)
        
        # Add export button
        self.export_button = QPushButton("Export Analysis")
        self.export_button.setToolTip("Export analysis with annotations")
        self.export_button.clicked.connect(self.export_analysis)
        self.export_button.setEnabled(False)
        top_layout.addWidget(self.export_button)
        
        main_layout.addLayout(top_layout)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Recordings list
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Recordings list
        recordings_group = QGroupBox("Recordings")
        recordings_layout = QVBoxLayout()
        
        self.recordings_list = QListWidget()
        self.recordings_list.itemSelectionChanged.connect(self.on_recording_selected)
        recordings_layout.addWidget(self.recordings_list)
        
        # Delete recording button
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_recording)
        recordings_layout.addWidget(self.delete_button)
        
        recordings_group.setLayout(recordings_layout)
        left_layout.addWidget(recordings_group)
        
        # Coaching tools
        coaching_group = QGroupBox("Coaching Tools")
        coaching_layout = QVBoxLayout()
        
        # Shot form scoring
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Stance:"), 0, 0)
        self.stance_score = QSpinBox()
        self.stance_score.setRange(1, 10)
        self.stance_score.setValue(5)
        form_layout.addWidget(self.stance_score, 0, 1)
        
        form_layout.addWidget(QLabel("Position:"), 1, 0)
        self.position_score = QSpinBox()
        self.position_score.setRange(1, 10)
        self.position_score.setValue(5)
        form_layout.addWidget(self.position_score, 1, 1)
        
        form_layout.addWidget(QLabel("Follow-through:"), 2, 0)
        self.follow_through_score = QSpinBox()
        self.follow_through_score.setRange(1, 10)
        self.follow_through_score.setValue(5)
        form_layout.addWidget(self.follow_through_score, 2, 1)
        
        # Overall evaluation
        form_layout.addWidget(QLabel("Overall:"), 3, 0)
        self.overall_score = QSpinBox()
        self.overall_score.setRange(1, 10)
        self.overall_score.setValue(5)
        form_layout.addWidget(self.overall_score, 3, 1)
        
        coaching_layout.addLayout(form_layout)
        
        # Annotation controls
        annotation_layout = QHBoxLayout()
        
        self.add_note_button = QPushButton("Add Note")
        self.add_note_button.clicked.connect(self.add_note)
        self.add_note_button.setEnabled(False)
        annotation_layout.addWidget(self.add_note_button)
        
        self.clear_annotations_button = QPushButton("Clear Annotations")
        self.clear_annotations_button.clicked.connect(self.clear_annotations)
        self.clear_annotations_button.setEnabled(False)
        annotation_layout.addWidget(self.clear_annotations_button)
        
        coaching_layout.addLayout(annotation_layout)
        
        # Save evaluation button
        self.save_evaluation_button = QPushButton("Save Evaluation")
        self.save_evaluation_button.clicked.connect(self.save_evaluation)
        self.save_evaluation_button.setEnabled(False)
        coaching_layout.addWidget(self.save_evaluation_button)
        
        coaching_group.setLayout(coaching_layout)
        left_layout.addWidget(coaching_group)
        
        left_panel.setLayout(left_layout)
        main_splitter.addWidget(left_panel)
        
        # Center panel: Video playback and controls
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        
        # Enhanced video display with annotation layer
        self.video_frame = VideoFrameWidget()
        center_layout.addWidget(self.video_frame)
        
        # Playback controls
        playback_controls = QHBoxLayout()
        
        # Frame step backward
        self.step_back_button = QPushButton("◀|")
        self.step_back_button.setToolTip("Previous frame")
        self.step_back_button.clicked.connect(self.step_backward)
        self.step_back_button.setEnabled(False)
        playback_controls.addWidget(self.step_back_button)
        
        # Play/pause button
        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Play/Pause")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        playback_controls.addWidget(self.play_button)
        
        # Frame step forward
        self.step_forward_button = QPushButton("|▶")
        self.step_forward_button.setToolTip("Next frame")
        self.step_forward_button.clicked.connect(self.step_forward)
        self.step_forward_button.setEnabled(False)
        playback_controls.addWidget(self.step_forward_button)
        
        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self.seek_playback)
        self.timeline_slider.sliderPressed.connect(self.pause_playback)
        playback_controls.addWidget(self.timeline_slider)
        
        # Time display
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(80)
        playback_controls.addWidget(self.time_label)
        
        # Mark shot frame button
        self.mark_shot_button = QPushButton("Mark Shot")
        self.mark_shot_button.setToolTip("Mark the current frame as the shot moment")
        self.mark_shot_button.clicked.connect(self.mark_shot_frame)
        self.mark_shot_button.setEnabled(False)
        playback_controls.addWidget(self.mark_shot_button)
        
        center_layout.addLayout(playback_controls)
        
        # Playback options
        options_group = QGroupBox("Playback Options")
        options_layout = QGridLayout()

        # Playback speed only (removing all overlay toggles)
        options_layout.addWidget(QLabel("Playback Speed:"), 0, 0)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(3)  # 1.0x is default
        self.speed_combo.currentIndexChanged.connect(self.update_playback_speed)
        options_layout.addWidget(self.speed_combo, 0, 1)

        options_group.setLayout(options_layout)
        center_layout.addWidget(options_group)
        
        center_panel.setLayout(center_layout)
        main_splitter.addWidget(center_panel)
        
        # Right panel: Analysis data
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Session info
        info_group = QGroupBox("Session Information")
        info_layout = QVBoxLayout()
        
        self.info_label = QLabel("No recording selected")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            background-color: #F5F8FA; 
            padding: 10px; 
            border-radius: 4px;
            min-height: 150px;
        """)
        info_layout.addWidget(self.info_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Shot history table
        shots_group = QGroupBox("Shot History")
        shots_layout = QVBoxLayout()

        self.shots_table = QTableWidget()
        self.shots_table.setColumnCount(5)
        self.shots_table.setHorizontalHeaderLabels([
            "Time", "Shot #", "Stability", "Follow-through", "Score"
        ])
        self.shots_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.shots_table.setAlternatingRowColors(True)
        self.shots_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #F5F8FA;
                gridline-color: #E1E8ED;
            }
        """)
        shots_layout.addWidget(self.shots_table)

        shots_group.setLayout(shots_layout)
        right_layout.addWidget(shots_group)
        # Metrics display    
        # Notes viewer
        notes_group = QGroupBox("Coach Notes")
        notes_layout = QVBoxLayout()
        
        self.notes_display = QTextEdit()
        self.notes_display.setReadOnly(True)
        self.notes_display.setMinimumHeight(150)
        self.notes_display.setStyleSheet("""
            background-color: #F5F8FA; 
            border: 1px solid #E1E8ED;
        """)
        notes_layout.addWidget(self.notes_display)
        
        notes_group.setLayout(notes_layout)
        right_layout.addWidget(notes_group)
        
        right_panel.setLayout(right_layout)
        main_splitter.addWidget(right_panel)
        
        # Set initial sizes
        main_splitter.setSizes([200, 650, 300])
        
        main_layout.addWidget(main_splitter)
        
        self.setLayout(main_layout)
    
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
        
        try:
            # Get sessions from database
            sessions = self.data_storage.get_sessions(self.user_id)
            
            # Add sessions to dropdown
            for session in sessions:
                session_text = f"{session['name']} ({session['created_at'][:10]})"
                self.session_selector.addItem(session_text, session['id'])
                
        except Exception as e:
            print(f"Error refreshing sessions: {e}")
    
    def set_session(self, session_id: int):
        """
        Set the current session.
        
        Args:
            session_id: ID of the session
        """
        if session_id <= 0:
            return
            
        # Update selector to match
        for i in range(self.session_selector.count()):
            if self.session_selector.itemData(i) == session_id:
                self.session_selector.setCurrentIndex(i)
                return
        
        # If not found in selector, set manually and load recordings
        self.session_id = session_id
        self.load_recordings()
    
    def on_session_changed(self, index: int):
        """
        Handle session selection change.
        
        Args:
            index: Index of the selected session in the dropdown
        """
        if index <= 0:  # "Select a session..." item
            self.session_id = None
            self.recordings_list.clear()
            return
        
        # Get session ID from combobox data
        session_id = self.session_selector.itemData(index)
        
        if session_id > 0:
            self.session_id = session_id
            self.load_recordings()
    
    def load_recordings(self):
        """Load recordings for the current user with enhanced error handling."""
        try:
            # Clear list
            self.recordings_list.clear()
            
            if not self.user_id:
                return
            
            # Check if directory exists
            user_recordings_dir = os.path.join("data/recordings", f"user_{self.user_id}")
            if not os.path.exists(user_recordings_dir):
                os.makedirs(user_recordings_dir, exist_ok=True)
            
            # Flag to filter by session
            filter_by_session = self.session_id is not None and self.session_id > 0
            
            # Get all recording metadata files
            recording_count = 0
            
            for filename in os.listdir(user_recordings_dir):
                if filename.endswith(".json"):
                    # Load metadata
                    metadata_path = os.path.join(user_recordings_dir, filename)
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Filter by session if applicable
                        if filter_by_session and metadata.get('session_id') != self.session_id:
                            continue
                        
                        # Check if video file exists
                        video_file = metadata.get('video_file', '')
                        video_path = os.path.join(user_recordings_dir, video_file)
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
                        recording_count += 1
                        
                    except Exception as e:
                        print(f"Error loading recording metadata: {str(e)}")
            
            # Update status if no recordings found
            if recording_count == 0:
                if filter_by_session:
                    no_rec_item = QListWidgetItem("No recordings for this session")
                else:
                    no_rec_item = QListWidgetItem("No recordings found")
                no_rec_item.setFlags(Qt.ItemFlag.NoItemFlags)
                self.recordings_list.addItem(no_rec_item)
                
        except Exception as e:
            print(f"Error loading recordings: {e}")
            import traceback
            traceback.print_exc()
    
    def on_recording_selected(self):
        """Handle recording selection change with enhanced error handling."""
        try:
            selected_items = self.recordings_list.selectedItems()
            
            if not selected_items or not selected_items[0].flags() & Qt.ItemFlag.ItemIsSelectable:
                self.clear_playback()
                return
            
            # Get selected recording metadata
            item = selected_items[0]
            metadata = item.data(Qt.ItemDataRole.UserRole)
            self.current_recording = metadata
            
            # Debug output
            print(f"Selected recording: {metadata.get('session_name')}, {metadata.get('timestamp')}")
            
            # Ensure user_recordings_dir is set and exists
            if not hasattr(self, 'user_recordings_dir') or not self.user_recordings_dir:
                self.user_recordings_dir = os.path.join("data/recordings", f"user_{self.user_id}")
                os.makedirs(self.user_recordings_dir, exist_ok=True)
            
            # Build and verify video path
            video_file = metadata.get('video_file', '')
            video_path = os.path.join(self.user_recordings_dir, video_file)
            
            if not video_file:
                self.video_frame.clear()
                self.video_frame.video_label.setText("Video filename missing in metadata")
                self.disable_playback_controls()
                return
                
            if not os.path.exists(video_path):
                self.video_frame.clear()
                self.video_frame.video_label.setText(f"Video file not found: {video_file}")
                self.disable_playback_controls()
                return
            
            # Initialize video capture with error handling
            try:
                self.playback_cap = cv2.VideoCapture(video_path)
                if not self.playback_cap.isOpened():
                    self.video_frame.clear()
                    self.video_frame.video_label.setText("Could not open video")
                    print(f"Failed to open video: {video_path}")
                    self.disable_playback_controls()
                    return
                    
                # Get video properties
                self.frame_rate = self.playback_cap.get(cv2.CAP_PROP_FPS)
                if self.frame_rate <= 0:
                    self.frame_rate = 30  # Default to 30 FPS if not detected
                    
                self.total_frames = int(self.playback_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.playback_duration = self.total_frames / self.frame_rate
                
                print(f"Video loaded: {self.frame_rate} FPS, {self.total_frames} frames, {self.playback_duration:.2f}s duration")
                
                # Update timeline slider
                self.timeline_slider.setMaximum(max(1, self.total_frames - 1))
                self.timeline_slider.setValue(0)
                self.current_frame = 0
                
                # Update time label
                minutes = int(self.playback_duration // 60)
                seconds = int(self.playback_duration % 60)
                self.time_label.setText(f"0:00 / {minutes}:{seconds:02d}")
                
                # Display first frame
                ret, frame = self.playback_cap.read()
                if ret:
                    # Process frame to remove unwanted overlays
                    frame = self.process_frame(frame)
                    self.video_frame.update_frame(frame)
                else:
                    self.video_frame.clear()
                    self.video_frame.video_label.setText("Could not read first frame")
                    print("Failed to read first frame")
                    self.disable_playback_controls()
                    return
                
                # Update metrics and info displays
                self.update_info_display(metadata)
                
                # Reset shot frame marker
                self.shot_frame_marked = False
                self.shot_frame = 0
                
                # Clear annotations
                self.video_frame.annotation_layer.clear_annotations()
                
                # Load notes if available
                self.load_notes(metadata)
                
                # Enable playback controls
                self.enable_playback_controls()
                
            except Exception as e:
                self.disable_playback_controls()
                self.video_frame.clear()
                self.video_frame.video_label.setText(f"Error opening video: {str(e)}")
                print(f"Error in video playback: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in recording selection: {e}")
            import traceback
            traceback.print_exc()
            self.disable_playback_controls()
            self.video_frame.clear()
    
    def clear_playback(self):
        """Clear all playback state and displays."""
        # Disable controls
        self.disable_playback_controls()
        
        # Clear displays
        self.video_frame.clear()
        self.info_label.setText("No recording selected")
        self.time_label.setText("0:00 / 0:00")
        self.stability_gauge.setValue(0)
        self.notes_display.clear()
        
        # Clear metrics
        for key in self.metric_labels:
            self.metric_labels[key].setText("--")
        
        # Stop playback if active
        if self.is_playing:
            self.pause_playback()
            
        # Release video capture
        if hasattr(self, 'playback_cap') and self.playback_cap is not None:
            self.playback_cap.release()
            self.playback_cap = None
        
        # Reset state
        self.current_recording = None
        self.current_frame = 0
        self.total_frames = 0
        self.shot_frame_marked = False
    
    def toggle_playback(self):
        """Toggle playback state between play and pause."""
        if not self.current_recording:
            return
        
        if not self.is_playing:
            self.start_playback()
        else:
            self.pause_playback()
    
    def start_playback(self):
        """Start video playback with error handling."""
        if not self.current_recording or not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
        
        try:
            # Update UI
            self.play_button.setText("⏸")
            self.play_button.setToolTip("Pause")
            self.is_playing = True
            
            # Calculate playback interval based on speed
            playback_speed = self.get_playback_speed()
            interval = int(1000 / (self.frame_rate * playback_speed))
            
            # Start timer
            self.playback_timer.start(interval)
            
        except Exception as e:
            print(f"Error starting playback: {e}")
            self.is_playing = False
            self.play_button.setText("▶")
    
    def pause_playback(self):
        """Pause video playback."""
        if not self.is_playing:
            return
        
        # Stop timer
        self.playback_timer.stop()
        
        # Update UI
        self.play_button.setText("▶")
        self.play_button.setToolTip("Play")
        self.is_playing = False
    
    def step_forward(self):
        """Step forward one frame."""
        if not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
            
        # Pause if playing
        if self.is_playing:
            self.pause_playback()
        
        # Read next frame
        ret, frame = self.playback_cap.read()
        
        if ret:
            # Update frame counter
            self.current_frame += 1
            
            # Process frame to remove unwanted overlays
            frame = self.process_frame(frame)
            
            # Update display
            self.video_frame.update_frame(frame)
            
            # Update timeline
            self.timeline_slider.setValue(self.current_frame)
            
            # Update time display
            self.update_time_display()
            
            # Update notes display if available
            self.update_notes_display()
        else:
            # End of video - seek back to last frame
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames - 1)
            self.current_frame = self.total_frames - 1
            self.timeline_slider.setValue(self.current_frame)
    
    def step_backward(self):
        """Step backward one frame."""
        if not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
            
        # Pause if playing
        if self.is_playing:
            self.pause_playback()
        
        # Calculate previous frame position
        prev_frame = max(0, self.current_frame - 1)
        
        # Set position
        self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
        self.current_frame = prev_frame
        
        # Read frame
        ret, frame = self.playback_cap.read()
        
        if ret:
            # Process frame to remove unwanted overlays
            frame = self.process_frame(frame)
            
            # Update display
            self.video_frame.update_frame(frame)
            
            # Update timeline
            self.timeline_slider.setValue(self.current_frame)
            
            # Update time display
            self.update_time_display()
            
            # Update notes display if available
            self.update_notes_display()
        else:
            print(f"Error reading frame at position {prev_frame}")
    
    def update_playback(self):
        """Update video playback with next frame and enhanced error handling."""
        if not self.is_playing or not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
        
        try:
            # Read next frame
            ret, frame = self.playback_cap.read()
            
            if not ret:
                # End of video
                self.pause_playback()
                self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.timeline_slider.setValue(0)
                self.current_frame = 0
                
                # Update time display
                self.update_time_display()
                
                # Read first frame again
                ret, frame = self.playback_cap.read()
                if ret:
                    # Process frame to remove unwanted overlays
                    frame = self.process_frame(frame)
                    self.video_frame.update_frame(frame)
                
                return
            
            # Update frame counter
            self.current_frame += 1
            
            # Process frame to remove unwanted overlays
            frame = self.process_frame(frame)
            
            # Update display
            self.video_frame.update_frame(frame)
            
            # Update timeline
            self.timeline_slider.setValue(self.current_frame)
            
            # Update time display
            self.update_time_display()
            
            # Update notes display if available
            self.update_notes_display()
            
        except Exception as e:
            print(f"Error updating playback: {e}")
            self.pause_playback()
    
    def seek_playback(self, position: int):
        """
        Seek to a specific position in the video with error handling.
        
        Args:
            position: Frame number to seek to
        """
        if not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
        
        try:
            # Validate position
            position = max(0, min(position, self.total_frames - 1))
            
            # Set position
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.current_frame = position
            
            # Read frame at new position
            ret, frame = self.playback_cap.read()
            
            if ret:
                # Process frame to remove unwanted overlays
                frame = self.process_frame(frame)
                
                # Update display
                self.video_frame.update_frame(frame)
                
                # Update time display
                self.update_time_display()
                
                # Update notes display
                self.update_notes_display()
                
                # Set position back by 1 so next read gets the correct frame
                self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            else:
                print(f"Error reading frame at position {position}")
                
        except Exception as e:
            print(f"Error seeking to position {position}: {e}")
    
    def update_time_display(self):
        """Update the time display based on current frame position."""
        if not hasattr(self, 'frame_rate') or self.frame_rate <= 0:
            return
            
        # Calculate times
        current_time = self.current_frame / self.frame_rate
        total_time = self.total_frames / self.frame_rate
        
        # Format as mm:ss
        current_minutes = int(current_time // 60)
        current_seconds = int(current_time % 60)
        
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        
        # Update label
        self.time_label.setText(
            f"{current_minutes}:{current_seconds:02d} / "
            f"{total_minutes}:{total_seconds:02d}"
        )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame - no overlay manipulation since video already has overlays.
        
        Args:
            frame: Original OpenCV frame
                
        Returns:
            Processed frame with only note markers if enabled
        """
        if frame is None:
            return None
        
        # Create a copy to avoid modifying the original
        processed_frame = frame.copy()
        
        # Add note markers if enabled and notes exist
        if hasattr(self, 'notes_checkbox') and self.notes_checkbox.isChecked() and hasattr(self, 'notes') and self.notes:
            processed_frame = self.add_note_markers(processed_frame)
        
        # Add shot frame marker if available
        if self.shot_frame_marked and self.current_frame == self.shot_frame:
            # Add a red border to indicate shot frame
            border_size = 10
            processed_frame = cv2.copyMakeBorder(
                processed_frame, 
                border_size, border_size, border_size, border_size, 
                cv2.BORDER_CONSTANT, 
                value=(0, 0, 255)  # Red border
            )
        
        return processed_frame
    
    
    def add_note_markers(self, frame: np.ndarray) -> np.ndarray:
        """
        Add note markers to the frame.
        
        Args:
            frame: Original frame
            
        Returns:
            Frame with note markers
        """
        # Check if we have notes for the current frame
        if not hasattr(self, 'notes') or self.current_frame not in self.notes:
            return frame
        
        # Add a note indicator
        h, w = frame.shape[:2]
        note_indicator = np.zeros((40, 150, 3), dtype=np.uint8)
        note_indicator[:, :] = (50, 120, 220)  # Orange background
        
        # Add text
        cv2.putText(note_indicator, "Coach Note", (10, 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Position at bottom left
        y_offset = h - 50
        x_offset = 10
        
        # Blend with frame
        roi = frame[y_offset:y_offset+40, x_offset:x_offset+150]
        alpha = 0.8
        cv2.addWeighted(note_indicator, alpha, roi, 1-alpha, 0, roi)
        frame[y_offset:y_offset+40, x_offset:x_offset+150] = roi
        
        return frame
    
    def get_playback_speed(self) -> float:
        """
        Get the current playback speed multiplier.
        
        Returns:
            Playback speed as float
        """
        speed_text = self.speed_combo.currentText()
        return float(speed_text.rstrip('x'))
    
    def update_playback_speed(self):
        """Update playback speed based on combobox selection."""
        if not self.is_playing:
            return
        
        # Get speed
        playback_speed = self.get_playback_speed()
        
        # Calculate new interval
        interval = int(1000 / (self.frame_rate * playback_speed))
        
        # Update timer
        self.playback_timer.start(interval)
    
    def update_overlays(self):
        """Update overlays based on checkbox states."""
        if not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
            
        # If playing, don't interrupt
        if self.is_playing:
            return
            
        # Refresh current frame
        current_pos = self.current_frame
        self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        ret, frame = self.playback_cap.read()
        if ret:
            # Process frame to update overlays
            frame = self.process_frame(frame)
            self.video_frame.update_frame(frame)
            
            # Reset position
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    def update_info_display(self, metadata: Dict):
        """
        Update the information display with just recording details, removing analysis summary.
        
        Args:
            metadata: Recording metadata dictionary
        """
        session_name = metadata.get('session_name', 'Unknown')
        timestamp = metadata.get('timestamp', 'Unknown')
        duration = metadata.get('duration', 0)
        
        # Format duration as mm:ss
        duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
        
        # Create HTML formatted info with only recording details
        info_text = f"""
        <h3>Recording Details</h3>
        <table style='width:100%;'>
            <tr><td><b>Session:</b></td><td>{session_name}</td></tr>
            <tr><td><b>Recorded:</b></td><td>{timestamp}</td></tr>
            <tr><td><b>Duration:</b></td><td>{duration_str}</td></tr>
        </table>
        """
        
        self.info_label.setText(info_text)
    
    
    def mark_shot_frame(self):
        """Mark the current frame in the recording as a shot."""
        if not hasattr(self, 'recording_metadata') or not hasattr(self, 'is_recording'):
            return
            
        # Get current time in recording
        elapsed = time.time() - self.recording_start_time - self.recording_paused_time
        
        # Add shot marker to metadata
        if 'shots' not in self.recording_metadata:
            self.recording_metadata['shots'] = []
            
        self.recording_metadata['shots'].append({
            'timestamp': elapsed,
            'frame_number': self.current_frame_number  # You may need to track this
        })
        
        # Also add current metrics with shot flag
        current_metrics = self._capture_current_metrics()
        if current_metrics:
            current_metrics['is_shot'] = True
            current_metrics['shot_time'] = elapsed
            self.recording_metadata['metrics'].append(current_metrics)
        
        # Show confirmation
        self.statusBar().showMessage("Shot marked at " + time.strftime("%M:%S", time.gmtime(elapsed)))

    def add_shot_note(self):
        """Add a note about the shot frame."""
        if not hasattr(self, 'notes'):
            self.notes = {}
            
        self.notes[self.shot_frame] = "Shot frame - crucial moment for follow-through analysis"
        self.update_notes_display()
    
    def add_note(self):
        """Add a coach's note at the current frame."""
        if not hasattr(self, 'playback_cap') or not self.playback_cap.isOpened():
            return
        
        # Create a dialog for the note
        note_dialog = QDialog(self)
        note_dialog.setWindowTitle("Add Coach Note")
        
        layout = QVBoxLayout()
        
        # Display current frame position
        frame_time = self.current_frame / self.frame_rate
        minutes = int(frame_time // 60)
        seconds = int(frame_time % 60)
        frame_label = QLabel(f"Frame: {self.current_frame} (Time: {minutes}:{seconds:02d})")
        layout.addWidget(frame_label)
        
        # Text area for note
        note_edit = QTextEdit()
        note_edit.setPlaceholderText("Enter coaching note for this frame...")
        layout.addWidget(note_edit)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(note_dialog.accept)
        button_box.rejected.connect(note_dialog.reject)
        layout.addWidget(button_box)
        
        note_dialog.setLayout(layout)
        
        # Show dialog
        if note_dialog.exec() == QDialog.DialogCode.Accepted:
            note_text = note_edit.toPlainText().strip()
            if note_text:
                # Initialize notes dict if not exists
                if not hasattr(self, 'notes'):
                    self.notes = {}
                
                # Add note for this frame
                self.notes[self.current_frame] = note_text
                
                # Update display
                self.update_notes_display()
                
                # Refresh frame to show note marker
                self.update_overlays()
    
    def update_notes_display(self):
        """Update the notes display for the current frame."""
        if not hasattr(self, 'notes') or not self.notes:
            self.notes_display.clear()
            return
        
        # Check if we have a note for the current frame
        note = self.notes.get(self.current_frame)
        
        if note:
            # Show the note for this frame
            html = f"""
            <html>
            <body>
                <h3>Note at {self.current_frame / self.frame_rate:.1f}s:</h3>
                <p>{note}</p>
            </body>
            </html>
            """
            self.notes_display.setHtml(html)
        else:
            # Show closest notes for context
            html = "<html><body><h3>Nearby Notes:</h3><ul>"
            
            nearby_notes = []
            for frame, text in self.notes.items():
                frame_diff = abs(frame - self.current_frame)
                if frame_diff < 30:  # Within ~1 second
                    nearby_notes.append((frame, text, frame_diff))
            
            # Sort by proximity
            nearby_notes.sort(key=lambda x: x[2])
            
            # Display up to 3 nearby notes
            for frame, text, _ in nearby_notes[:3]:
                frame_time = frame / self.frame_rate
                minutes = int(frame_time // 60)
                seconds = int(frame_time % 60)
                
                direction = "before" if frame < self.current_frame else "after"
                
                html += f"<li><b>At {minutes}:{seconds:02d}</b> ({direction}): {text}</li>"
            
            html += "</ul></body></html>"
            self.notes_display.setHtml(html)
    
    def clear_annotations(self):
        """Clear all annotations from the current recording."""
        if hasattr(self, 'video_frame') and hasattr(self.video_frame, 'annotation_layer'):
            self.video_frame.annotation_layer.clear_annotations()
    
    def load_notes(self, metadata: Dict):
        """
        Load notes from metadata if available.
        
        Args:
            metadata: Recording metadata dictionary
        """
        # Reset notes
        self.notes = {}
        
        # Check if metadata has notes
        if 'notes' in metadata and isinstance(metadata['notes'], dict):
            # Convert string frame numbers to integers
            for frame_str, note in metadata['notes'].items():
                try:
                    frame = int(frame_str)
                    self.notes[frame] = note
                except ValueError:
                    continue
        
        # Update notes display
        self.notes_display.clear()
    
    def save_evaluation(self):
        """Save the coach's evaluation of the recording."""
        if not self.current_recording:
            return
            
        # Get evaluation scores
        stance = self.stance_score.value()
        position = self.position_score.value()
        follow_through = self.follow_through_score.value()
        overall = self.overall_score.value()
        
        # Prepare evaluation data
        evaluation = {
            "stance": stance,
            "position": position,
            "follow_through": follow_through,
            "overall": overall,
            "notes": self.notes if hasattr(self, 'notes') else {},
            "shot_frame": self.shot_frame if self.shot_frame_marked else None,
            "evaluated_at": datetime.datetime.now().isoformat()
        }
        
        # Get metadata path
        if not hasattr(self, 'user_recordings_dir') or not self.current_recording:
            return
            
        video_file = self.current_recording.get('video_file', '')
        metadata_path = os.path.join(
            self.user_recordings_dir, 
            os.path.splitext(video_file)[0] + ".json"
        )
        
        # Update metadata
        if os.path.exists(metadata_path):
            try:
                # Load existing metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Add evaluation
                metadata['evaluation'] = evaluation
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # Show confirmation
                QMessageBox.information(
                    self, "Evaluation Saved", 
                    "Coaching evaluation has been saved successfully."
                )
                
            except Exception as e:
                print(f"Error saving evaluation: {e}")
                QMessageBox.critical(
                    self, "Error", 
                    f"Failed to save evaluation: {str(e)}"
                )
    
    def export_analysis(self):
        """Export the analysis with annotations and notes."""
        if not self.current_recording:
            return
            
        # Get export format
        export_format = "PDF"  # Could add options in UI
        
        # Get export path
        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis", 
            f"shooting_analysis_{datetime.datetime.now().strftime('%Y%m%d')}", 
            "PDF Files (*.pdf);;Image Files (*.png)"
        )
        
        if not export_path:
            return
            
        # Show a message about export
        QMessageBox.information(
            self, "Export Feature", 
            "This feature would export the full analysis with:\n"
            "- Key frame screenshots\n"
            "- Stability metrics\n"
            "- Coach's notes and annotations\n"
            "- Recommendations for improvement\n\n"
            "Export functionality would be implemented based on specific coaching requirements."
        )
    
    def delete_recording(self):
        """Delete the selected recording."""
        selected_items = self.recordings_list.selectedItems()
        
        if not selected_items or not selected_items[0].flags() & Qt.ItemFlag.ItemIsSelectable:
            return
        
        # Get selected recording metadata
        item = selected_items[0]
        metadata = item.data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Recording", 
            "Are you sure you want to delete this recording? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Stop playback if this recording
        if self.current_recording == metadata:
            self.pause_playback()
            if hasattr(self, 'playback_cap') and self.playback_cap is not None:
                self.playback_cap.release()
                self.playback_cap = None
            
            self.clear_playback()
        
        # Delete files
        try:
            video_file = metadata.get('video_file', '')
            video_path = os.path.join(self.user_recordings_dir, video_file)
            
            metadata_path = os.path.join(
                self.user_recordings_dir, 
                os.path.splitext(video_file)[0] + ".json"
            )
            
            # Delete files if they exist
            if os.path.exists(video_path):
                os.remove(video_path)
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Remove from list
            row = self.recordings_list.row(item)
            self.recordings_list.takeItem(row)
            
            # Show confirmation
            QMessageBox.information(
                self, "Recording Deleted", 
                "The recording has been deleted successfully."
            )
            
        except Exception as e:
            print(f"Error deleting recording: {e}")
            QMessageBox.critical(
                self, "Error", 
                f"Failed to delete recording: {str(e)}"
            )
    
    def enable_playback_controls(self):
        """Enable playback controls."""
        self.play_button.setEnabled(True)
        self.step_forward_button.setEnabled(True)
        self.step_back_button.setEnabled(True)
        self.mark_shot_button.setEnabled(True)
        self.mark_shot_button.setText("Mark Shot")
        self.add_note_button.setEnabled(True)
        self.clear_annotations_button.setEnabled(True)
        self.save_evaluation_button.setEnabled(True)
        self.export_button.setEnabled(True)
    
    def disable_playback_controls(self):
        """Disable playback controls."""
        self.play_button.setEnabled(False)
        self.step_forward_button.setEnabled(False)
        self.step_back_button.setEnabled(False)
        self.mark_shot_button.setEnabled(False)
        self.add_note_button.setEnabled(False)
        self.clear_annotations_button.setEnabled(False)
        self.save_evaluation_button.setEnabled(False)
        self.export_button.setEnabled(False)
    
    def closeEvent(self, event):
        """Handle widget close event with proper cleanup."""
        # Stop playback if active
        if self.is_playing:
            self.pause_playback()
        
        # Clean up resources
        if hasattr(self, 'playback_cap') and self.playback_cap is not None:
            self.playback_cap.release()
            self.playback_cap = None
        
        # Accept the event
        event.accept()
    
    def create_default_metadata(self, session_id, video_file):
        """Create default metadata if file is missing or corrupt."""
        now = datetime.datetime.now()
        return {
            "user_id": self.user_id,
            "session_id": session_id,
            "session_name": f"Session {session_id}",
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "video_file": video_file,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "metrics": [],
            "duration": 0,
            "paused_time": 0
        }
    
    def update_shots_table(self, metadata: Dict):
        """
        Update the shots history table with data from recording metadata.
        
        Args:
            metadata: Recording metadata dictionary
        """
        self.shots_table.setRowCount(0)  # Clear table
        
        # Check if we have shots data
        shots = metadata.get('shots', [])
        metrics = metadata.get('metrics', [])
        
        if not shots and metrics:
            # No explicit shot markers, but we can use metrics with is_shot flag
            shots = [m for m in metrics if m.get('is_shot', False)]
        
        if not shots and metrics:
            # Still no shots, use every 30th frame as a sample point
            sample_interval = 30
            shots = [m for i, m in enumerate(metrics) if i % sample_interval == 0]
        
        # If we have shots or metrics, populate the table
        if shots:
            for i, shot in enumerate(shots):
                row_position = self.shots_table.rowCount()
                self.shots_table.insertRow(row_position)
                
                # Time column
                shot_time = shot.get('timestamp', 0)
                minutes = int(shot_time // 60)
                seconds = int(shot_time % 60)
                time_str = f"{minutes}:{seconds:02d}"
                self.shots_table.setItem(row_position, 0, QTableWidgetItem(time_str))
                
                # Shot number
                self.shots_table.setItem(row_position, 1, QTableWidgetItem(str(i + 1)))
                
                # Find metrics for this shot
                shot_metrics = shot
                if 'timestamp' in shot and metrics:
                    # Try to find matching metrics by timestamp
                    shot_time = shot['timestamp']
                    closest_metric = min(metrics, key=lambda m: abs(m.get('timestamp', 0) - shot_time))
                    if abs(closest_metric.get('timestamp', 0) - shot_time) < 1.0:  # Within 1 second
                        shot_metrics = closest_metric
                
                # Stability
                stability = self._calculate_stability_score(shot_metrics)
                stability_item = QTableWidgetItem(f"{stability:.1f}%")
                self.shots_table.setItem(row_position, 2, stability_item)
                
                # Follow-through
                follow_through = shot_metrics.get('follow_through_score', 0)
                follow_item = QTableWidgetItem(f"{follow_through:.2f}")
                self.shots_table.setItem(row_position, 3, follow_item)
                
                # Score (if available)
                score = shot_metrics.get('shot_score', '-')
                self.shots_table.setItem(row_position, 4, QTableWidgetItem(str(score)))
    
    