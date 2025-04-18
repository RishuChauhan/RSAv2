from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QSlider, QDoubleSpinBox, QComboBox,
    QCheckBox, QTabWidget, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings

from typing import Dict, List, Optional
import json
import os

from src.data_storage import DataStorage

class SettingsWidget(QWidget):
    """
    Widget for application settings and configuration.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the settings widget.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        
        # Initialize QSettings
        self.settings = QSettings("RifleShootingAnalysis", "App")
        
        # Initialize UI
        self.init_ui()
        
        # Load settings
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface elements."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Settings")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # Create tab widget for different settings categories
        tab_widget = QTabWidget()
        
        # Camera and Audio tab
        camera_widget = self.create_camera_settings()
        tab_widget.addTab(camera_widget, "Camera & Audio")
        
        # Algorithm Parameters tab
        algorithm_widget = self.create_algorithm_settings()
        tab_widget.addTab(algorithm_widget, "Algorithm Parameters")
        
        # User Interface tab
        ui_widget = self.create_ui_settings()
        tab_widget.addTab(ui_widget, "User Interface")
        
        # Data Management tab
        data_widget = self.create_data_settings()
        tab_widget.addTab(data_widget, "Data Management")
        
        main_layout.addWidget(tab_widget)
        
        # Save/Cancel buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        button_layout.addWidget(self.reset_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.load_settings)
        button_layout.addWidget(self.cancel_button)
        
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def create_camera_settings(self) -> QWidget:
        """
        Create camera and audio settings widget.
        
        Returns:
            Settings widget
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_grid = QGridLayout()
        
        # Camera selection
        camera_grid.addWidget(QLabel("Camera Device:"), 0, 0)
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("Default Camera", 0)
        self.camera_selector.addItem("Camera 1", 1)
        self.camera_selector.addItem("Camera 2", 2)
        camera_grid.addWidget(self.camera_selector, 0, 1)
        
        # Resolution
        camera_grid.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_selector = QComboBox()
        self.resolution_selector.addItem("1280x720", "1280x720")
        self.resolution_selector.addItem("1920x1080", "1920x1080")
        self.resolution_selector.addItem("640x480", "640x480")
        camera_grid.addWidget(self.resolution_selector, 1, 1)
        
        # Frame rate
        camera_grid.addWidget(QLabel("Frame Rate:"), 2, 0)
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setMinimum(15)
        self.fps_spinner.setMaximum(60)
        self.fps_spinner.setValue(30)
        camera_grid.addWidget(self.fps_spinner, 2, 1)
        
        camera_group.setLayout(camera_grid)
        layout.addWidget(camera_group)
        
        # Audio settings
        audio_group = QGroupBox("Audio Settings")
        audio_grid = QGridLayout()
        
        # Audio device
        audio_grid.addWidget(QLabel("Audio Device:"), 0, 0)
        self.audio_selector = QComboBox()
        self.audio_selector.addItem("Default Microphone", 0)
        self.audio_selector.addItem("Microphone 1", 1)
        self.audio_selector.addItem("Microphone 2", 2)
        audio_grid.addWidget(self.audio_selector, 0, 1)
        
        # Shot detection threshold
        audio_grid.addWidget(QLabel("Shot Detection Threshold:"), 1, 0)
        threshold_layout = QHBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_spinner.setValue(v / 100)
        )
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_spinner = QDoubleSpinBox()
        self.threshold_spinner.setMinimum(0.0)
        self.threshold_spinner.setMaximum(1.0)
        self.threshold_spinner.setSingleStep(0.01)
        self.threshold_spinner.setValue(0.5)
        self.threshold_spinner.valueChanged.connect(
            lambda v: self.threshold_slider.setValue(int(v * 100))
        )
        threshold_layout.addWidget(self.threshold_spinner)
        
        audio_grid.addLayout(threshold_layout, 1, 1)
        
        # Enable/disable audio detection
        audio_grid.addWidget(QLabel("Enable Audio Detection:"), 2, 0)
        self.audio_enabled = QCheckBox()
        self.audio_enabled.setChecked(True)
        audio_grid.addWidget(self.audio_enabled, 2, 1)
        
        audio_group.setLayout(audio_grid)
        layout.addWidget(audio_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_algorithm_settings(self) -> QWidget:
        """
        Create algorithm parameter settings widget.
        
        Returns:
            Settings widget
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Joint tracking settings
        tracking_group = QGroupBox("Joint Tracking")
        tracking_grid = QGridLayout()
        
        # Detection confidence
        tracking_grid.addWidget(QLabel("Detection Confidence:"), 0, 0)
        self.detection_confidence = QDoubleSpinBox()
        self.detection_confidence.setMinimum(0.1)
        self.detection_confidence.setMaximum(1.0)
        self.detection_confidence.setSingleStep(0.05)
        self.detection_confidence.setValue(0.5)
        tracking_grid.addWidget(self.detection_confidence, 0, 1)
        
        # Tracking confidence
        tracking_grid.addWidget(QLabel("Tracking Confidence:"), 1, 0)
        self.tracking_confidence = QDoubleSpinBox()
        self.tracking_confidence.setMinimum(0.1)
        self.tracking_confidence.setMaximum(1.0)
        self.tracking_confidence.setSingleStep(0.05)
        self.tracking_confidence.setValue(0.5)
        tracking_grid.addWidget(self.tracking_confidence, 1, 1)
        
        # Model complexity
        tracking_grid.addWidget(QLabel("Model Complexity:"), 2, 0)
        self.model_complexity = QComboBox()
        self.model_complexity.addItem("Lite (Faster)", 0)
        self.model_complexity.addItem("Standard", 1)
        self.model_complexity.addItem("Heavy (More Accurate)", 2)
        self.model_complexity.setCurrentIndex(2)
        tracking_grid.addWidget(self.model_complexity, 2, 1)
        
        tracking_group.setLayout(tracking_grid)
        layout.addWidget(tracking_group)
        
        # Stability metrics settings
        metrics_group = QGroupBox("Stability Metrics")
        metrics_grid = QGridLayout()
        
        # Joint weights
        metrics_grid.addWidget(QLabel("Joint Weights:"), 0, 0)
        weights_layout = QGridLayout()
        
        # Shoulder weight
        weights_layout.addWidget(QLabel("Shoulders:"), 0, 0)
        self.shoulder_weight = QDoubleSpinBox()
        self.shoulder_weight.setMinimum(0.0)
        self.shoulder_weight.setMaximum(1.0)
        self.shoulder_weight.setSingleStep(0.1)
        self.shoulder_weight.setValue(0.2)
        weights_layout.addWidget(self.shoulder_weight, 0, 1)
        
        # Elbow weight
        weights_layout.addWidget(QLabel("Elbows:"), 1, 0)
        self.elbow_weight = QDoubleSpinBox()
        self.elbow_weight.setMinimum(0.0)
        self.elbow_weight.setMaximum(1.0)
        self.elbow_weight.setSingleStep(0.1)
        self.elbow_weight.setValue(0.3)
        weights_layout.addWidget(self.elbow_weight, 1, 1)
        
        # Wrist weight
        weights_layout.addWidget(QLabel("Wrists:"), 2, 0)
        self.wrist_weight = QDoubleSpinBox()
        self.wrist_weight.setMinimum(0.0)
        self.wrist_weight.setMaximum(1.0)
        self.wrist_weight.setSingleStep(0.1)
        self.wrist_weight.setValue(0.4)
        weights_layout.addWidget(self.wrist_weight, 2, 1)
        
        # Nose weight
        weights_layout.addWidget(QLabel("Nose:"), 3, 0)
        self.nose_weight = QDoubleSpinBox()
        self.nose_weight.setMinimum(0.0)
        self.nose_weight.setMaximum(1.0)
        self.nose_weight.setSingleStep(0.1)
        self.nose_weight.setValue(0.1)
        weights_layout.addWidget(self.nose_weight, 3, 1)
        
        metrics_grid.addLayout(weights_layout, 0, 1)
        
        # Lambda parameters
        metrics_grid.addWidget(QLabel("Lambda Parameters:"), 1, 0)
        lambda_layout = QGridLayout()
        
        # Lambda sway
        lambda_layout.addWidget(QLabel("λ₁ (Sway):"), 0, 0)
        self.lambda_sway = QDoubleSpinBox()
        self.lambda_sway.setMinimum(0.1)
        self.lambda_sway.setMaximum(1.0)
        self.lambda_sway.setSingleStep(0.1)
        self.lambda_sway.setValue(0.5)
        lambda_layout.addWidget(self.lambda_sway, 0, 1)
        
        # Lambda deviation
        lambda_layout.addWidget(QLabel("λ₂ (Deviation):"), 1, 0)
        self.lambda_dev = QDoubleSpinBox()
        self.lambda_dev.setMinimum(0.1)
        self.lambda_dev.setMaximum(1.0)
        self.lambda_dev.setSingleStep(0.1)
        self.lambda_dev.setValue(0.5)
        lambda_layout.addWidget(self.lambda_dev, 1, 1)
        
        metrics_grid.addLayout(lambda_layout, 1, 1)
        
        metrics_group.setLayout(metrics_grid)
        layout.addWidget(metrics_group)
        
        # Fuzzy logic settings
        fuzzy_group = QGroupBox("Fuzzy Logic Feedback")
        fuzzy_grid = QGridLayout()
        
        # Sway thresholds
        fuzzy_grid.addWidget(QLabel("Sway Thresholds (mm/s):"), 0, 0)
        sway_layout = QGridLayout()
        
        # Low-medium boundary
        sway_layout.addWidget(QLabel("Low-Medium:"), 0, 0)
        self.sway_low_med = QDoubleSpinBox()
        self.sway_low_med.setMinimum(1.0)
        self.sway_low_med.setMaximum(20.0)
        self.sway_low_med.setSingleStep(0.5)
        self.sway_low_med.setValue(5.0)
        sway_layout.addWidget(self.sway_low_med, 0, 1)
        
        # Medium-high boundary
        sway_layout.addWidget(QLabel("Medium-High:"), 1, 0)
        self.sway_med_high = QDoubleSpinBox()
        self.sway_med_high.setMinimum(5.0)
        self.sway_med_high.setMaximum(30.0)
        self.sway_med_high.setSingleStep(0.5)
        self.sway_med_high.setValue(10.0)
        sway_layout.addWidget(self.sway_med_high, 1, 1)
        
        fuzzy_grid.addLayout(sway_layout, 0, 1)
        
        # Stability thresholds
        fuzzy_grid.addWidget(QLabel("Stability Thresholds:"), 1, 0)
        stability_layout = QGridLayout()
        
        # Low-medium boundary
        stability_layout.addWidget(QLabel("Low-Medium:"), 0, 0)
        self.stability_low_med = QDoubleSpinBox()
        self.stability_low_med.setMinimum(0.2)
        self.stability_low_med.setMaximum(0.6)
        self.stability_low_med.setSingleStep(0.05)
        self.stability_low_med.setValue(0.4)
        stability_layout.addWidget(self.stability_low_med, 0, 1)
        
        # Medium-high boundary
        stability_layout.addWidget(QLabel("Medium-High:"), 1, 0)
        self.stability_med_high = QDoubleSpinBox()
        self.stability_med_high.setMinimum(0.5)
        self.stability_med_high.setMaximum(0.9)
        self.stability_med_high.setSingleStep(0.05)
        self.stability_med_high.setValue(0.7)
        stability_layout.addWidget(self.stability_med_high, 1, 1)
        
        fuzzy_grid.addLayout(stability_layout, 1, 1)
        
        fuzzy_group.setLayout(fuzzy_grid)
        layout.addWidget(fuzzy_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_ui_settings(self) -> QWidget:
        """
        Create user interface settings widget.
        
        Returns:
            Settings widget
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Display settings
        display_group = QGroupBox("Display Settings")
        display_grid = QGridLayout()
        
        # Skeleton display
        display_grid.addWidget(QLabel("Show Skeleton:"), 0, 0)
        self.show_skeleton = QCheckBox()
        self.show_skeleton.setChecked(True)
        display_grid.addWidget(self.show_skeleton, 0, 1)
        
        # Heatmap display
        display_grid.addWidget(QLabel("Show Heatmap:"), 1, 0)
        self.show_heatmap = QCheckBox()
        self.show_heatmap.setChecked(True)
        display_grid.addWidget(self.show_heatmap, 1, 1)
        
        # Metrics display
        display_grid.addWidget(QLabel("Show Metrics:"), 2, 0)
        self.show_metrics = QCheckBox()
        self.show_metrics.setChecked(True)
        display_grid.addWidget(self.show_metrics, 2, 1)
        
        # Feedback display
        display_grid.addWidget(QLabel("Show Feedback:"), 3, 0)
        self.show_feedback = QCheckBox()
        self.show_feedback.setChecked(True)
        display_grid.addWidget(self.show_feedback, 3, 1)
        
        display_group.setLayout(display_grid)
        layout.addWidget(display_group)
        
        # Visualization settings
        viz_group = QGroupBox("3D Visualization")
        viz_grid = QGridLayout()
        
        # Default view
        viz_grid.addWidget(QLabel("Default View:"), 0, 0)
        self.default_view = QComboBox()
        self.default_view.addItems(["Front", "Side", "Top", "Custom"])
        self.default_view.setCurrentIndex(0)
        viz_grid.addWidget(self.default_view, 0, 1)
        
        # Custom elevation
        viz_grid.addWidget(QLabel("Custom Elevation:"), 1, 0)
        elevation_layout = QHBoxLayout()
        
        self.elevation_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_slider.setMinimum(-90)
        self.elevation_slider.setMaximum(90)
        self.elevation_slider.setValue(20)
        self.elevation_slider.valueChanged.connect(
            lambda v: self.elevation_spinner.setValue(v)
        )
        elevation_layout.addWidget(self.elevation_slider)
        
        self.elevation_spinner = QSpinBox()
        self.elevation_spinner.setMinimum(-90)
        self.elevation_spinner.setMaximum(90)
        self.elevation_spinner.setValue(20)
        self.elevation_spinner.valueChanged.connect(
            lambda v: self.elevation_slider.setValue(v)
        )
        elevation_layout.addWidget(self.elevation_spinner)
        
        viz_grid.addLayout(elevation_layout, 1, 1)
        
        # Custom azimuth
        viz_grid.addWidget(QLabel("Custom Azimuth:"), 2, 0)
        azimuth_layout = QHBoxLayout()
        
        self.azimuth_slider = QSlider(Qt.Orientation.Horizontal)
        self.azimuth_slider.setMinimum(-180)
        self.azimuth_slider.setMaximum(180)
        self.azimuth_slider.setValue(-60)
        self.azimuth_slider.valueChanged.connect(
            lambda v: self.azimuth_spinner.setValue(v)
        )
        azimuth_layout.addWidget(self.azimuth_slider)
        
        self.azimuth_spinner = QSpinBox()
        self.azimuth_spinner.setMinimum(-180)
        self.azimuth_spinner.setMaximum(180)
        self.azimuth_spinner.setValue(-60)
        self.azimuth_spinner.valueChanged.connect(
            lambda v: self.azimuth_slider.setValue(v)
        )
        azimuth_layout.addWidget(self.azimuth_spinner)
        
        viz_grid.addLayout(azimuth_layout, 2, 1)
        
        viz_group.setLayout(viz_grid)
        layout.addWidget(viz_group)
        
        # Notification settings
        notification_group = QGroupBox("Notifications")
        notification_grid = QGridLayout()
        
        # Sound notifications
        notification_grid.addWidget(QLabel("Sound Notifications:"), 0, 0)
        self.sound_notifications = QCheckBox()
        self.sound_notifications.setChecked(True)
        notification_grid.addWidget(self.sound_notifications, 0, 1)
        
        # Shot detection notification
        notification_grid.addWidget(QLabel("Shot Detection Alert:"), 1, 0)
        self.shot_alert = QCheckBox()
        self.shot_alert.setChecked(True)
        notification_grid.addWidget(self.shot_alert, 1, 1)
        
        # Session completion notification
        notification_grid.addWidget(QLabel("Session Completion Alert:"), 2, 0)
        self.session_alert = QCheckBox()
        self.session_alert.setChecked(True)
        notification_grid.addWidget(self.session_alert, 2, 1)
        
        notification_group.setLayout(notification_grid)
        layout.addWidget(notification_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_data_settings(self) -> QWidget:
        """
        Create data management settings widget.
        
        Returns:
            Settings widget
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Session settings
        session_group = QGroupBox("Session Settings")
        session_grid = QGridLayout()
        
        # Minimum shot count
        session_grid.addWidget(QLabel("Minimum Shots per Session:"), 0, 0)
        self.min_shots = QSpinBox()
        self.min_shots.setMinimum(1)
        self.min_shots.setMaximum(50)
        self.min_shots.setValue(10)
        session_grid.addWidget(self.min_shots, 0, 1)
        
        # Auto-save interval
        session_grid.addWidget(QLabel("Auto-save Interval (minutes):"), 1, 0)
        self.autosave_interval = QSpinBox()
        self.autosave_interval.setMinimum(1)
        self.autosave_interval.setMaximum(60)
        self.autosave_interval.setValue(5)
        session_grid.addWidget(self.autosave_interval, 1, 1)
        
        # Auto-capture settings
        session_grid.addWidget(QLabel("Auto-capture Shots:"), 2, 0)
        self.auto_capture = QCheckBox()
        self.auto_capture.setChecked(True)
        session_grid.addWidget(self.auto_capture, 2, 1)
        
        session_group.setLayout(session_grid)
        layout.addWidget(session_group)
        
        # Data storage settings
        storage_group = QGroupBox("Data Storage")
        storage_grid = QGridLayout()
        
        # Database location
        storage_grid.addWidget(QLabel("Database Location:"), 0, 0)
        db_layout = QHBoxLayout()
        
        self.db_path = QLabel("data/rifle_shot_analysis.db")
        db_layout.addWidget(self.db_path)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_db_location)
        db_layout.addWidget(self.browse_button)
        
        storage_grid.addLayout(db_layout, 0, 1)
        
        # Recording storage
        storage_grid.addWidget(QLabel("Recording Storage:"), 1, 0)
        rec_layout = QHBoxLayout()
        
        self.rec_path = QLabel("data/recordings")
        rec_layout.addWidget(self.rec_path)
        
        self.rec_browse_button = QPushButton("Browse...")
        self.rec_browse_button.clicked.connect(self.browse_rec_location)
        rec_layout.addWidget(self.rec_browse_button)
        
        storage_grid.addLayout(rec_layout, 1, 1)
        
        # Data export
        storage_grid.addWidget(QLabel("Data Export Format:"), 2, 0)
        self.export_format = QComboBox()
        self.export_format.addItems(["CSV", "JSON", "Excel"])
        self.export_format.setCurrentIndex(0)
        storage_grid.addWidget(self.export_format, 2, 1)
        
        # Data management buttons
        storage_grid.addWidget(QLabel("Data Management:"), 3, 0)
        data_buttons = QHBoxLayout()
        
        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_data)
        data_buttons.addWidget(self.export_button)
        
        self.backup_button = QPushButton("Backup Database")
        self.backup_button.clicked.connect(self.backup_database)
        data_buttons.addWidget(self.backup_button)
        
        storage_grid.addLayout(data_buttons, 3, 1)
        
        storage_group.setLayout(storage_grid)
        layout.addWidget(storage_group)
        
        # User management
        user_group = QGroupBox("User Management")
        user_grid = QGridLayout()
        
        # Change password
        self.change_password_button = QPushButton("Change Password")
        self.change_password_button.clicked.connect(self.change_password)
        user_grid.addWidget(self.change_password_button, 0, 0)
        
        # Delete account
        self.delete_account_button = QPushButton("Delete Account")
        self.delete_account_button.clicked.connect(self.delete_account)
        self.delete_account_button.setStyleSheet("background-color: #ff5555;")
        user_grid.addWidget(self.delete_account_button, 0, 1)
        
        user_group.setLayout(user_grid)
        layout.addWidget(user_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def set_user(self, user_id: int):
        """
        Set the current user.
        
        Args:
            user_id: ID of the current user
        """
        self.user_id = user_id
        
        # Load user-specific settings
        self.settings.beginGroup(f"user_{user_id}")
        self.load_settings()
        self.settings.endGroup()
    
    def load_settings(self):
        """Load settings from QSettings."""
        # Camera & Audio settings
        self.camera_selector.setCurrentIndex(self.settings.value("camera/device", 0, int))
        resolution_idx = self.resolution_selector.findData(self.settings.value("camera/resolution", "1280x720"))
        self.resolution_selector.setCurrentIndex(resolution_idx if resolution_idx >= 0 else 0)
        self.fps_spinner.setValue(self.settings.value("camera/fps", 30, int))
        
        self.audio_selector.setCurrentIndex(self.settings.value("audio/device", 0, int))
        self.threshold_spinner.setValue(self.settings.value("audio/threshold", 0.5, float))
        self.threshold_slider.setValue(int(self.threshold_spinner.value() * 100))
        self.audio_enabled.setChecked(self.settings.value("audio/enabled", True, bool))
        
        # Algorithm settings
        self.detection_confidence.setValue(self.settings.value("tracking/detection_confidence", 0.5, float))
        self.tracking_confidence.setValue(self.settings.value("tracking/tracking_confidence", 0.5, float))
        self.model_complexity.setCurrentIndex(self.settings.value("tracking/model_complexity", 2, int))
        
        self.shoulder_weight.setValue(self.settings.value("metrics/shoulder_weight", 0.2, float))
        self.elbow_weight.setValue(self.settings.value("metrics/elbow_weight", 0.3, float))
        self.wrist_weight.setValue(self.settings.value("metrics/wrist_weight", 0.4, float))
        self.nose_weight.setValue(self.settings.value("metrics/nose_weight", 0.1, float))
        
        self.lambda_sway.setValue(self.settings.value("metrics/lambda_sway", 0.5, float))
        self.lambda_dev.setValue(self.settings.value("metrics/lambda_dev", 0.5, float))
        
        self.sway_low_med.setValue(self.settings.value("fuzzy/sway_low_med", 5.0, float))
        self.sway_med_high.setValue(self.settings.value("fuzzy/sway_med_high", 10.0, float))
        self.stability_low_med.setValue(self.settings.value("fuzzy/stability_low_med", 0.4, float))
        self.stability_med_high.setValue(self.settings.value("fuzzy/stability_med_high", 0.7, float))
        
        # UI settings
        self.show_skeleton.setChecked(self.settings.value("ui/show_skeleton", True, bool))
        self.show_heatmap.setChecked(self.settings.value("ui/show_heatmap", True, bool))
        self.show_metrics.setChecked(self.settings.value("ui/show_metrics", True, bool))
        self.show_feedback.setChecked(self.settings.value("ui/show_feedback", True, bool))
        
        view_idx = self.default_view.findText(self.settings.value("ui/default_view", "Front"))
        self.default_view.setCurrentIndex(view_idx if view_idx >= 0 else 0)
        self.elevation_spinner.setValue(self.settings.value("ui/elevation", 20, int))
        self.azimuth_spinner.setValue(self.settings.value("ui/azimuth", -60, int))
        
        self.sound_notifications.setChecked(self.settings.value("ui/sound_notifications", True, bool))
        self.shot_alert.setChecked(self.settings.value("ui/shot_alert", True, bool))
        self.session_alert.setChecked(self.settings.value("ui/session_alert", True, bool))
        
        # Data settings
        self.min_shots.setValue(self.settings.value("data/min_shots", 10, int))
        self.autosave_interval.setValue(self.settings.value("data/autosave_interval", 5, int))
        self.auto_capture.setChecked(self.settings.value("data/auto_capture", True, bool))
        
        self.db_path.setText(self.settings.value("data/db_path", "data/rifle_shot_analysis.db"))
        self.rec_path.setText(self.settings.value("data/rec_path", "data/recordings"))
        
        format_idx = self.export_format.findText(self.settings.value("data/export_format", "CSV"))
        self.export_format.setCurrentIndex(format_idx if format_idx >= 0 else 0)
    
    def save_settings(self):
        """Save settings to QSettings."""
        # Begin group for current user if set
        if self.user_id:
            self.settings.beginGroup(f"user_{self.user_id}")
        
        # Camera & Audio settings
        self.settings.setValue("camera/device", self.camera_selector.currentIndex())
        self.settings.setValue("camera/resolution", self.resolution_selector.currentData())
        self.settings.setValue("camera/fps", self.fps_spinner.value())
        
        self.settings.setValue("audio/device", self.audio_selector.currentIndex())
        self.settings.setValue("audio/threshold", self.threshold_spinner.value())
        self.settings.setValue("audio/enabled", self.audio_enabled.isChecked())
        
        # Algorithm settings
        self.settings.setValue("tracking/detection_confidence", self.detection_confidence.value())
        self.settings.setValue("tracking/tracking_confidence", self.tracking_confidence.value())
        self.settings.setValue("tracking/model_complexity", self.model_complexity.currentIndex())
        
        self.settings.setValue("metrics/shoulder_weight", self.shoulder_weight.value())
        self.settings.setValue("metrics/elbow_weight", self.elbow_weight.value())
        self.settings.setValue("metrics/wrist_weight", self.wrist_weight.value())
        self.settings.setValue("metrics/nose_weight", self.nose_weight.value())
        
        self.settings.setValue("metrics/lambda_sway", self.lambda_sway.value())
        self.settings.setValue("metrics/lambda_dev", self.lambda_dev.value())
        
        self.settings.setValue("fuzzy/sway_low_med", self.sway_low_med.value())
        self.settings.setValue("fuzzy/sway_med_high", self.sway_med_high.value())
        self.settings.setValue("fuzzy/stability_low_med", self.stability_low_med.value())
        self.settings.setValue("fuzzy/stability_med_high", self.stability_med_high.value())
        
        # UI settings
        self.settings.setValue("ui/show_skeleton", self.show_skeleton.isChecked())
        self.settings.setValue("ui/show_heatmap", self.show_heatmap.isChecked())
        self.settings.setValue("ui/show_metrics", self.show_metrics.isChecked())
        self.settings.setValue("ui/show_feedback", self.show_feedback.isChecked())
        
        self.settings.setValue("ui/default_view", self.default_view.currentText())
        self.settings.setValue("ui/elevation", self.elevation_spinner.value())
        self.settings.setValue("ui/azimuth", self.azimuth_spinner.value())
        
        self.settings.setValue("ui/sound_notifications", self.sound_notifications.isChecked())
        self.settings.setValue("ui/shot_alert", self.shot_alert.isChecked())
        self.settings.setValue("ui/session_alert", self.session_alert.isChecked())
        
        # Data settings
        self.settings.setValue("data/min_shots", self.min_shots.value())
        self.settings.setValue("data/autosave_interval", self.autosave_interval.value())
        self.settings.setValue("data/auto_capture", self.auto_capture.isChecked())
        
        self.settings.setValue("data/db_path", self.db_path.text())
        self.settings.setValue("data/rec_path", self.rec_path.text())
        
        self.settings.setValue("data/export_format", self.export_format.currentText())
        
        # End group if using user-specific settings
        if self.user_id:
            self.settings.endGroup()
        
        QMessageBox.information(self, "Settings", "Settings saved successfully.")
    
    def reset_settings(self):
        """Reset settings to default values."""
        # Confirm reset
        reply = QMessageBox.question(
            self, "Reset Settings", 
            "Are you sure you want to reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Clear user-specific settings
        if self.user_id:
            self.settings.beginGroup(f"user_{self.user_id}")
            self.settings.remove("")  # Remove all keys in group
            self.settings.endGroup()
        
        # Load default values
        self.load_settings()
        
        QMessageBox.information(self, "Settings", "Settings reset to defaults.")
    
    def browse_db_location(self):
        """Browse for database location."""
        from PyQt6.QtWidgets import QFileDialog
        
        # Get current path
        current_path = self.db_path.text()
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Database Location", 
            current_path, "SQLite Database (*.db)"
        )
        
        if file_path:
            self.db_path.setText(file_path)
    
    def browse_rec_location(self):
        """Browse for recordings location."""
        from PyQt6.QtWidgets import QFileDialog
        
        # Get current path
        current_path = self.rec_path.text()
        
        # Open directory dialog
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Recordings Directory", current_path
        )
        
        if dir_path:
            self.rec_path.setText(dir_path)
    
    def export_data(self):
        """Export user data."""
        if not self.user_id:
            QMessageBox.warning(self, "Export Data", "No user is currently active.")
            return
        
        # Get export format
        export_format = self.export_format.currentText()
        
        from PyQt6.QtWidgets import QFileDialog
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", 
            f"rifle_shooting_data_{self.user_id}", 
            f"{export_format} Files (*.{export_format.lower()})"
        )
        
        if not file_path:
            return
        
        # Placeholder for actual export functionality
        QMessageBox.information(
            self, "Export Data", 
            f"Data would be exported to {file_path} in {export_format} format.\n"
            "This functionality is not implemented in this demo."
        )
    
    def backup_database(self):
        """Backup the database."""
        from PyQt6.QtWidgets import QFileDialog
        import shutil
        
        # Get current database path
        db_path = self.db_path.text()
        
        # Check if database exists
        if not os.path.exists(db_path):
            QMessageBox.warning(self, "Backup Database", "Database file not found.")
            return
        
        # Get backup location
        backup_path, _ = QFileDialog.getSaveFileName(
            self, "Backup Database", 
            f"{os.path.splitext(db_path)[0]}_backup.db", 
            "SQLite Database (*.db)"
        )
        
        if not backup_path:
            return
        
        try:
            # Copy database file
            shutil.copy2(db_path, backup_path)
            
            QMessageBox.information(
                self, "Backup Database", 
                f"Database successfully backed up to:\n{backup_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Backup Error", 
                f"Failed to backup database: {str(e)}"
            )
    
    def change_password(self):
        """Change user password."""
        if not self.user_id:
            QMessageBox.warning(self, "Change Password", "No user is currently active.")
            return
        
        # This would typically involve:
        # 1. Asking for current password
        # 2. Asking for new password (with confirmation)
        # 3. Updating the password in the database
        
        QMessageBox.information(
            self, "Change Password", 
            "Password change functionality is not implemented in this demo."
        )
    
    def delete_account(self):
        """Delete user account."""
        if not self.user_id:
            QMessageBox.warning(self, "Delete Account", "No user is currently active.")
            return
        
        # Confirm deletion
        reply = QMessageBox.warning(
            self, "Delete Account", 
            "Are you sure you want to delete your account? This cannot be undone!\n"
            "All your data will be permanently deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # This would typically involve:
        # 1. Deleting user data from database
        # 2. Deleting user recordings
        # 3. Logging out user
        
        QMessageBox.information(
            self, "Delete Account", 
            "Account deletion functionality is not implemented in this demo."
        )