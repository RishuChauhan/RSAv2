from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt, QSize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from typing import Dict, List, Optional

from src.data_storage import DataStorage

class VisualizationWidget(QWidget):
    """
    Widget for 3D visualization of joint positions and movement.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the 3D visualization widget.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        self.session_id = None
        
        # Currently selected shots
        self.selected_shots = []
        
        # Joint visibility settings
        self.joint_visibility = {
            'SHOULDERS': True,
            'ELBOWS': True,
            'WRISTS': True,
            'NOSE': True,
            'HIPS': True,
            'ANKLES': False
        }
        
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
        
        # Spacer
        control_layout.addStretch()
        
        # Reset view button
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_3d_view)
        control_layout.addWidget(self.reset_view_button)
        
        main_layout.addLayout(control_layout)
        
        # Main content splitter (shots list, 3D view, and options)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Shots list
        shots_widget = QWidget()
        shots_layout = QVBoxLayout()
        shots_layout.addWidget(QLabel("Select Shots to Visualize:"))
        
        self.shots_list = QListWidget()
        self.shots_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.shots_list.itemSelectionChanged.connect(self.on_shot_selection_changed)
        shots_layout.addWidget(self.shots_list)
        
        # Compare button
        self.compare_button = QPushButton("Compare Selected Shots")
        self.compare_button.clicked.connect(self.compare_shots)
        shots_layout.addWidget(self.compare_button)
        
        shots_widget.setLayout(shots_layout)
        main_splitter.addWidget(shots_widget)
        
        # Middle: 3D visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        
        # Create 3D figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Initial plot setup
        self.setup_3d_plot()
        
        viz_layout.addWidget(self.canvas)
        
        viz_widget.setLayout(viz_layout)
        main_splitter.addWidget(viz_widget)
        
        # Right side: Options
        options_widget = QWidget()
        options_layout = QVBoxLayout()
        
        # Visualization options
        options_group = QGroupBox("Visualization Options")
        options_grid = QGridLayout()
        
        # Joint visibility checkboxes
        row = 0
        self.joint_checkboxes = {}
        
        for joint, visible in self.joint_visibility.items():
            checkbox = QCheckBox(joint)
            checkbox.setChecked(visible)
            checkbox.stateChanged.connect(self.on_joint_visibility_changed)
            options_grid.addWidget(checkbox, row, 0)
            self.joint_checkboxes[joint] = checkbox
            row += 1
        
        # Visualization type options
        options_grid.addWidget(QLabel("Visualization Type:"), row, 0)
        row += 1
        
        self.viz_type_selector = QComboBox()
        self.viz_type_selector.addItems([
            "Points and Lines",
            "Motion Trails",
            "Heatmap"
        ])
        self.viz_type_selector.currentIndexChanged.connect(self.update_visualization)
        options_grid.addWidget(self.viz_type_selector, row, 0)
        
        options_group.setLayout(options_grid)
        options_layout.addWidget(options_group)
        
        # Shot information
        info_group = QGroupBox("Shot Information")
        info_layout = QVBoxLayout()
        
        self.shot_info_label = QLabel("Select a shot to see details.")
        self.shot_info_label.setWordWrap(True)
        info_layout.addWidget(self.shot_info_label)
        
        info_group.setLayout(info_layout)
        options_layout.addWidget(info_group)
        
        # Add stretch to bottom
        options_layout.addStretch()
        
        options_widget.setLayout(options_layout)
        main_splitter.addWidget(options_widget)
        
        # Set the initial sizes
        main_splitter.setSizes([200, 600, 200])
        
        main_layout.addWidget(main_splitter)
        
        self.setLayout(main_layout)
    
    def setup_3d_plot(self):
        """Set up the initial 3D plot."""
        self.ax.clear()
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')  # In MediaPipe, Z is depth
        self.ax.set_zlabel('Y')  # In PyQt, Y is inverted (downward)
        
        # Set initial view
        self.ax.view_init(elev=20, azim=-60)
        
        # Set axis limits (will be adjusted based on data)
        self.ax.set_xlim3d([0, 640])
        self.ax.set_ylim3d([-100, 100])
        self.ax.set_zlim3d([0, 480])
        
        # Set title
        self.ax.set_title('3D Joint Visualization')
        
        # Add grid
        self.ax.grid(True)
        
        # Draw canvas
        self.figure.tight_layout()
        self.canvas.draw()
    
    def reset_3d_view(self):
        """Reset the 3D view to the default perspective."""
        self.ax.view_init(elev=20, azim=-60)
        self.canvas.draw()
    
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
    
    def set_session(self, session_id: int):
        """
        Set the current session and load shots.
        
        Args:
            session_id: ID of the session
        """
        # Update selector to match
        for i in range(self.session_selector.count()):
            if self.session_selector.itemData(i) == session_id:
                self.session_selector.setCurrentIndex(i)
                return
        
        # If not found in selector, set manually and load
        self.session_id = session_id
        self.load_session_shots()
    
    def on_session_changed(self, index: int):
        """
        Handle session selection change.
        
        Args:
            index: Index of the selected session in the dropdown
        """
        if index <= 0:  # "Select a session..." item
            self.session_id = None
            self.shots_list.clear()
            self.setup_3d_plot()  # Reset plot
            return
        
        # Get session ID from combobox data
        session_id = self.session_selector.itemData(index)
        
        if session_id > 0:
            self.session_id = session_id
            self.load_session_shots()
    
    def load_session_shots(self):
        """Load shots for the current session."""
        if not self.session_id:
            return
        
        # Clear current shots
        self.shots_list.clear()
        
        # Get shots from database
        shots = self.data_storage.get_shots(self.session_id)
        
        # Add shots to list
        for i, shot in enumerate(shots):
            # Format timestamp
            timestamp = shot['timestamp'].split('T')[1][:8]  # HH:MM:SS
            
            # Create list item
            item_text = f"Shot {i+1} - {timestamp} - Score: {shot['subjective_score']}"
            item = QListWidgetItem(item_text)
            
            # Store shot data as item data
            item.setData(Qt.ItemDataRole.UserRole, shot)
            
            self.shots_list.addItem(item)
    
    def on_shot_selection_changed(self):
        """Handle shot selection changes."""
        selected_items = self.shots_list.selectedItems()
        
        if not selected_items:
            self.shot_info_label.setText("Select a shot to see details.")
            return
        
        # Get selected shots
        self.selected_shots = []
        for item in selected_items:
            shot = item.data(Qt.ItemDataRole.UserRole)
            self.selected_shots.append(shot)
        
        # If only one shot selected, show detailed info
        if len(self.selected_shots) == 1:
            shot = self.selected_shots[0]
            
            # Format shot info
            info_text = f"Shot ID: {shot['id']}\n"
            info_text += f"Time: {shot['timestamp']}\n"
            info_text += f"Subjective Score: {shot['subjective_score']}\n\n"
            
            # Add metrics
            metrics = shot['metrics']
            if 'follow_through_score' in metrics:
                info_text += f"Follow-through: {metrics['follow_through_score']:.2f}\n"
            
            if 'sway_velocity' in metrics:
                avg_sway = sum(metrics['sway_velocity'].values()) / len(metrics['sway_velocity'])
                info_text += f"Avg Sway: {avg_sway:.2f} mm/s\n"
            
            self.shot_info_label.setText(info_text)
        else:
            # Multiple shots selected
            self.shot_info_label.setText(f"{len(self.selected_shots)} shots selected.\nClick 'Compare' to visualize.")
        
        # Update visualization if automatic update enabled
        self.update_visualization()
    
    def on_joint_visibility_changed(self):
        """Handle joint visibility checkbox changes."""
        # Update visibility settings
        for joint, checkbox in self.joint_checkboxes.items():
            self.joint_visibility[joint] = checkbox.isChecked()
        
        # Update visualization
        self.update_visualization()
    
    def update_visualization(self):
        """Update the 3D visualization based on selected shots and options."""
        if not self.selected_shots:
            self.setup_3d_plot()  # Reset plot
            return
        
        # Clear the plot
        self.ax.clear()
        
        # Get visualization type
        viz_type = self.viz_type_selector.currentText()
        
        # Color map for multiple shots
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        
        # Joint connections for skeleton
        connections = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('NOSE', 'LEFT_SHOULDER'),
            ('NOSE', 'RIGHT_SHOULDER'),
            ('LEFT_HIP', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_ANKLE')
        ]
        
        # Process each selected shot
        for shot_idx, shot in enumerate(self.selected_shots):
            color = colors[shot_idx % len(colors)]
            
            # Extract joint positions from metrics
            if 'joint_positions' not in shot['metrics']:
                # If joint positions not directly stored, try to reconstruct from other metrics
                continue
            
            joint_positions = shot['metrics']['joint_positions']
            
            # Plot based on visualization type
            if viz_type == "Points and Lines":
                self.plot_skeleton(joint_positions, connections, color, shot_idx)
            elif viz_type == "Motion Trails":
                self.plot_motion_trails(joint_positions, color, shot_idx)
            elif viz_type == "Heatmap":
                self.plot_heatmap(joint_positions, shot_idx)
        
        # Set plot properties
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')
        self.ax.set_title('3D Joint Visualization')
        self.ax.grid(True)
        
        # Draw the updated plot
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_skeleton(self, joint_positions, connections, color, shot_idx):
        """
        Plot the skeleton with points and lines.
        
        Args:
            joint_positions: Dictionary of joint positions
            connections: List of joint connections
            color: Color for this skeleton
            shot_idx: Index of the shot
        """
        # Track min/max coordinates for proper scaling
        x_vals, y_vals, z_vals = [], [], []
        
        # Plot joints as points
        for joint, visible in self.joint_visibility.items():
            if not visible or joint not in joint_positions:
                continue
            
            pos = joint_positions[joint]
            
            # In case we have multiple samples for the joint
            if isinstance(pos, list):
                # Just use the first position for skeleton
                pos = pos[0]
            
            x, y, z = pos['x'], pos['z'], pos['y']  # Note the coordinate mapping
            
            # Add to coordinate lists for scaling
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            
            # Plot the joint
            self.ax.scatter([x], [y], [z], color=color, s=50, 
                            label=f"{joint} (Shot {shot_idx+1})" if shot_idx == 0 else "")
        
        # Plot connections as lines
        for joint1, joint2 in connections:
            if (joint1 not in self.joint_visibility or joint2 not in self.joint_visibility or
                not self.joint_visibility[joint1] or not self.joint_visibility[joint2] or
                joint1 not in joint_positions or joint2 not in joint_positions):
                continue
            
            pos1 = joint_positions[joint1]
            pos2 = joint_positions[joint2]
            
            # In case we have multiple samples
            if isinstance(pos1, list):
                pos1 = pos1[0]
            if isinstance(pos2, list):
                pos2 = pos2[0]
            
            x1, y1, z1 = pos1['x'], pos1['z'], pos1['y']
            x2, y2, z2 = pos2['x'], pos2['z'], pos2['y']
            
            self.ax.plot([x1, x2], [y1, y2], [z1, z2], color=color)
        
        # Adjust axis limits if we have data
        if x_vals and y_vals and z_vals:
            x_range = max(x_vals) - min(x_vals)
            y_range = max(y_vals) - min(y_vals)
            z_range = max(z_vals) - min(z_vals)
            
            # Add padding
            padding = max(x_range, y_range, z_range) * 0.2
            
            self.ax.set_xlim3d([min(x_vals) - padding, max(x_vals) + padding])
            self.ax.set_ylim3d([min(y_vals) - padding, max(y_vals) + padding])
            self.ax.set_zlim3d([min(z_vals) - padding, max(z_vals) + padding])
    
    def plot_motion_trails(self, joint_positions, color, shot_idx):
        """
        Plot motion trails for joints.
        
        Args:
            joint_positions: Dictionary of joint positions
            color: Color for this shot
            shot_idx: Index of the shot
        """
        # This is a placeholder implementation, as the actual joint position history
        # would need to be stored in the shot data for this to work properly
        
        # For demonstration, we'll just plot the skeleton
        self.plot_skeleton(joint_positions, [], color, shot_idx)
        
        # If we had actual motion data, we would plot trails like this:
        """
        for joint, visible in self.joint_visibility.items():
            if not visible or joint not in joint_positions:
                continue
            
            positions = joint_positions[joint]
            
            # If we have a history of positions
            if isinstance(positions, list) and len(positions) > 1:
                x = [pos['x'] for pos in positions]
                y = [pos['z'] for pos in positions]  # Z in MediaPipe is depth
                z = [pos['y'] for pos in positions]  # Y in PyQt is inverted
                
                # Plot the trail with decreasing alpha
                for i in range(1, len(x)):
                    alpha = 0.3 + 0.7 * (i / len(x))  # Fade from 0.3 to 1.0
                    self.ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], color=color, alpha=alpha)
                
                # Plot the final position
                self.ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=50,
                                label=f"{joint} (Shot {shot_idx+1})" if shot_idx == 0 else "")
        """
    
    def plot_heatmap(self, joint_positions, shot_idx):
        """
        Plot heatmap visualization of joint density.
        
        Args:
            joint_positions: Dictionary of joint positions
            shot_idx: Index of the shot
        """
        # This is a placeholder implementation, as a proper heatmap
        # would require more data points than we typically have in a single shot
        
        # For demonstration, we'll just plot the skeleton with color-coded stability
        # based on sway velocity
        
        # Get sway velocities if available
        sway_velocities = self.selected_shots[shot_idx]['metrics'].get('sway_velocity', {})
        
        # Plot joints as points with color based on sway
        for joint, visible in self.joint_visibility.items():
            if not visible or joint not in joint_positions:
                continue
            
            pos = joint_positions[joint]
            
            # In case we have multiple samples
            if isinstance(pos, list):
                pos = pos[0]
            
            x, y, z = pos['x'], pos['z'], pos['y']
            
            # Get sway velocity for this joint
            sway = sway_velocities.get(joint, 0)
            
            # Color mapping: green (stable) to red (unstable)
            # Higher sway = more red
            normalized_sway = min(1.0, sway / 20.0)  # Cap at 20 mm/s
            color = [normalized_sway, 1.0 - normalized_sway, 0.0]
            
            # Plot the joint
            self.ax.scatter([x], [y], [z], color=color, s=100,
                           label=f"{joint} (Shot {shot_idx+1})" if shot_idx == 0 else "")
    
    def compare_shots(self):
        """Compare multiple selected shots in the 3D visualization."""
        if len(self.selected_shots) < 2:
            # Need at least 2 shots to compare
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Compare Shots", 
                               "Please select at least 2 shots to compare.")
            return
        
        # Just trigger the visualization update
        self.update_visualization()