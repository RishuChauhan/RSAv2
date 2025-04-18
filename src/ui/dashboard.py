from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from typing import Dict, List, Optional
import numpy as np

from src.data_storage import DataStorage

class DashboardWidget(QWidget):
    """
    Dashboard widget displaying session data, statistics, and trends.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the dashboard widget.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        self.current_session = None
        
        # Initialize UI
        self.init_ui()
        
        # Refresh timer (update every 5 seconds)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)
    
    def init_ui(self):
        """Initialize the user interface elements."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Dashboard")
        title_label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        main_layout.addWidget(title_label)
        
        # Session selector
        session_layout = QHBoxLayout()
        session_layout.addWidget(QLabel("Select Session:"))
        
        self.session_selector = QComboBox()
        self.session_selector.setMinimumWidth(300)
        self.session_selector.currentIndexChanged.connect(self.on_session_changed)
        session_layout.addWidget(self.session_selector)
        
        session_layout.addStretch()
        
        # Session stats at a glance
        self.session_stats_label = QLabel("No session selected")
        session_layout.addWidget(self.session_stats_label)
        
        main_layout.addLayout(session_layout)
        
        # Create splitter for tables and charts
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Shot data table
        self.shots_table = QTableWidget()
        self.shots_table.setColumnCount(6)
        self.shots_table.setHorizontalHeaderLabels([
            "Shot #", "Timestamp", "Subjective Score", 
            "Follow-through", "Sway Velocity", "Postural Stability"
        ])
        self.shots_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        splitter.addWidget(self.shots_table)
        
        # Charts widget
        charts_widget = QWidget()
        charts_layout = QHBoxLayout()
        charts_widget.setLayout(charts_layout)
        
        # Score trend chart
        self.score_fig = Figure(figsize=(5, 4), dpi=100)
        self.score_canvas = FigureCanvas(self.score_fig)
        charts_layout.addWidget(self.score_canvas)
        
        # Stability metrics chart
        self.metrics_fig = Figure(figsize=(5, 4), dpi=100)
        self.metrics_canvas = FigureCanvas(self.metrics_fig)
        charts_layout.addWidget(self.metrics_canvas)
        
        splitter.addWidget(charts_widget)
        
        # Set the initial sizes of the splitter
        splitter.setSizes([300, 400])
        
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
    
    def set_user(self, user_id: int):
        """
        Set the current user and load their sessions.
        
        Args:
            user_id: ID of the current user
        """
        self.user_id = user_id
        self.refresh_sessions()
    
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
    
    def on_session_changed(self, index: int):
        """
        Handle session selection change.
        
        Args:
            index: Index of the selected session in the dropdown
        """
        if index <= 0:  # "Select a session..." item
            self.current_session = None
            self.session_stats_label.setText("No session selected")
            self.clear_data()
            return
        
        # Get session ID from combobox data
        session_id = self.session_selector.itemData(index)
        
        if session_id > 0:
            self.current_session = session_id
            self.refresh_data()
    
    def refresh_data(self):
        """Refresh all dashboard data for the current session."""
        if not self.current_session:
            return
        
        # Get session statistics
        stats = self.data_storage.get_session_stats(self.current_session)
        
        # Update stats label
        stats_text = f"Shots: {stats['shot_count']} | "
        stats_text += f"Avg Score: {stats['avg_subjective_score']:.1f} | "
        stats_text += f"Best: {stats['max_subjective_score']} | "
        stats_text += f"Worst: {stats['min_subjective_score']}"
        
        self.session_stats_label.setText(stats_text)
        
        # Get shot data
        shots = self.data_storage.get_shots(self.current_session)
        
        # Update table
        self.update_shots_table(shots)
        
        # Update charts
        self.update_score_chart(shots)
        self.update_metrics_chart(shots)
    
    def update_shots_table(self, shots: List[Dict]):
        """
        Update the shots table with shot data.
        
        Args:
            shots: List of shot dictionaries
        """
        self.shots_table.setRowCount(0)  # Clear table
        
        for i, shot in enumerate(shots):
            row_position = self.shots_table.rowCount()
            self.shots_table.insertRow(row_position)
            
            # Shot number
            self.shots_table.setItem(row_position, 0, QTableWidgetItem(str(i + 1)))
            
            # Timestamp (just show time part)
            timestamp = shot['timestamp'].split('T')[1][:8]  # HH:MM:SS
            self.shots_table.setItem(row_position, 1, QTableWidgetItem(timestamp))
            
            # Subjective score
            self.shots_table.setItem(row_position, 2, 
                                    QTableWidgetItem(str(shot['subjective_score'])))
            
            # Follow-through score
            follow_through = shot['metrics'].get('follow_through_score', 0)
            follow_through_item = QTableWidgetItem(f"{follow_through:.2f}")
            self.shots_table.setItem(row_position, 3, follow_through_item)
            
            # Average sway velocity (of upper body)
            sway_velocities = shot['metrics'].get('sway_velocity', {})
            if sway_velocities:
                upper_body_sway = 0
                count = 0
                
                for joint in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
                             'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'NOSE']:
                    if joint in sway_velocities:
                        upper_body_sway += sway_velocities[joint]
                        count += 1
                
                avg_sway = upper_body_sway / count if count > 0 else 0
                sway_item = QTableWidgetItem(f"{avg_sway:.2f} mm/s")
            else:
                sway_item = QTableWidgetItem("N/A")
                
            self.shots_table.setItem(row_position, 4, sway_item)
            
            # Postural stability (average of DevX and DevY)
            dev_x = shot['metrics'].get('dev_x', {}).get('UPPER_BODY', 0)
            dev_y = shot['metrics'].get('dev_y', {}).get('UPPER_BODY', 0)
            
            stability = (dev_x + dev_y) / 2
            stability_item = QTableWidgetItem(f"{stability:.2f} px")
            self.shots_table.setItem(row_position, 5, stability_item)
    
    def update_score_chart(self, shots: List[Dict]):
        """
        Update the score trend chart.
        
        Args:
            shots: List of shot dictionaries
        """
        # Clear figure
        self.score_fig.clear()
        
        if not shots:
            self.score_canvas.draw()
            return
        
        # Prepare data
        shot_numbers = list(range(1, len(shots) + 1))
        subjective_scores = [shot['subjective_score'] for shot in shots]
        follow_through_scores = [shot['metrics'].get('follow_through_score', 0) * 10 for shot in shots]
        
        # Create subplot
        ax = self.score_fig.add_subplot(111)
        
        # Plot data
        ax.plot(shot_numbers, subjective_scores, 'b-', marker='o', label='Subjective Score')
        ax.plot(shot_numbers, follow_through_scores, 'g--', marker='s', label='Follow-through (×10)')
        
        # Set labels and title
        ax.set_xlabel('Shot Number')
        ax.set_ylabel('Score')
        ax.set_title('Score Trends')
        
        # Set y-axis limits
        ax.set_ylim(0, 11)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Draw canvas
        self.score_fig.tight_layout()
        self.score_canvas.draw()
    
    def update_metrics_chart(self, shots: List[Dict]):
        """
        Update the stability metrics chart.
        
        Args:
            shots: List of shot dictionaries
        """
        # Clear figure
        self.metrics_fig.clear()
        
        if not shots:
            self.metrics_canvas.draw()
            return
        
        # Prepare data
        shot_numbers = list(range(1, len(shots) + 1))
        
        # Extract sway velocity data for different body parts
        wrist_sway = []
        elbow_sway = []
        shoulder_sway = []
        
        for shot in shots:
            sway_velocities = shot['metrics'].get('sway_velocity', {})
            
            # Average wrist sway
            left_wrist = sway_velocities.get('LEFT_WRIST', 0)
            right_wrist = sway_velocities.get('RIGHT_WRIST', 0)
            wrist_sway.append((left_wrist + right_wrist) / 2)
            
            # Average elbow sway
            left_elbow = sway_velocities.get('LEFT_ELBOW', 0)
            right_elbow = sway_velocities.get('RIGHT_ELBOW', 0)
            elbow_sway.append((left_elbow + right_elbow) / 2)
            
            # Average shoulder sway
            left_shoulder = sway_velocities.get('LEFT_SHOULDER', 0)
            right_shoulder = sway_velocities.get('RIGHT_SHOULDER', 0)
            shoulder_sway.append((left_shoulder + right_shoulder) / 2)
        
        # Create subplot
        ax = self.metrics_fig.add_subplot(111)
        
        # Plot data
        ax.plot(shot_numbers, wrist_sway, 'r-', marker='o', label='Wrist Sway')
        ax.plot(shot_numbers, elbow_sway, 'g-', marker='s', label='Elbow Sway')
        ax.plot(shot_numbers, shoulder_sway, 'b-', marker='^', label='Shoulder Sway')
        
        # Set labels and title
        ax.set_xlabel('Shot Number')
        ax.set_ylabel('Sway Velocity (mm/s)')
        ax.set_title('Joint Stability Metrics')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Draw canvas
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()
    
    def clear_data(self):
        """Clear all data displays when no session is selected."""
        # Clear table
        self.shots_table.setRowCount(0)
        
        # Clear charts
        self.score_fig.clear()
        self.score_canvas.draw()
        
        self.metrics_fig.clear()
        self.metrics_canvas.draw()