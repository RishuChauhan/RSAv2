from __future__ import print_function
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QGridLayout, QCheckBox, QSizePolicy, QSlider
)
from PyQt6.QtCore import Qt, QSize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import traceback
import sys
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
        
        # Flag to track if the widget is closing - add this BEFORE calling any methods
        self._is_closing = False
        
        # Currently selected shots
        self.selected_shots = []

        self.skeleton_rotation_angle = 0 
        # Joint visibility settings
        self.joint_visibility = {
            'SHOULDERS': True,
            'ELBOWS': True,
            'WRISTS': True,
            'NOSE': True,
            'HIPS': True,
            'ANKLES': False
        }
        
        # Initialize matplotlib objects
        self._create_matplotlib_figure()
        
        # Then initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface elements with improved 3D visualization."""
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
        
        # Add rotation buttons that will rotate the data
        self.front_view_button = QPushButton("Front View")
        self.front_view_button.clicked.connect(lambda: self.rotate_skeleton(0))
        control_layout.addWidget(self.front_view_button)
        
        self.left_view_button = QPushButton("Left View")
        self.left_view_button.clicked.connect(lambda: self.rotate_skeleton(90))
        control_layout.addWidget(self.left_view_button)
        
        self.back_view_button = QPushButton("Back View")
        self.back_view_button.clicked.connect(lambda: self.rotate_skeleton(180))
        control_layout.addWidget(self.back_view_button)
        
        self.right_view_button = QPushButton("Right View")
        self.right_view_button.clicked.connect(lambda: self.rotate_skeleton(270))
        control_layout.addWidget(self.right_view_button)
        
        main_layout.addLayout(control_layout)
        
        # Main content splitter (shots list, 3D view, and options)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Shots list
        shots_widget = QWidget()
        shots_layout = QVBoxLayout()
        shots_layout.addWidget(QLabel("Select Shot(s) to Visualize:"))
        
        self.shots_list = QListWidget()
        self.shots_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.shots_list.itemSelectionChanged.connect(self.on_shot_selection_changed)
        shots_layout.addWidget(self.shots_list)
        
        # Compare button (optional now)
        self.compare_button = QPushButton("Compare Selected Shots")
        self.compare_button.clicked.connect(self.compare_shots)
        self.compare_button.setToolTip("Optional: Select multiple shots and click to compare them")
        shots_layout.addWidget(self.compare_button)
        
        shots_widget.setLayout(shots_layout)
        main_splitter.addWidget(shots_widget)

       # Middle: 3D visualization 
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()

        viz_layout.addWidget(self.canvas, stretch=3)

        self.canvas_parent_layout = viz_layout
        self.old_canvas = self.canvas

        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        self.canvas.setMinimumHeight(400)  # Set a minimum height
        

        
        # Add coordinate display with reduced size
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Coordinates:"))
        
        # Create a scrollable text area for coordinates
        from PyQt6.QtWidgets import QTextEdit
        self.coord_label = QTextEdit()
        self.coord_label.setReadOnly(True)
        self.coord_label.setStyleSheet("""
            background-color: #E3F2FD;
            padding: 3px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 8px;
            max-height: 80px;
        """)
        self.coord_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, 
            QSizePolicy.Policy.Maximum
        )
        self.coord_label.setMaximumHeight(80)  # Set maximum height
        coord_layout.addWidget(self.coord_label)
        viz_layout.addLayout(coord_layout, stretch=1)  # Use smaller stretch factor
        
        # Complete the layout setup
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
        
        # Visualization type options (but with only "Points and Lines" enabled)
        options_grid.addWidget(QLabel("Visualization Type:"), row, 0)
        row += 1
        
        self.viz_type_selector = QComboBox()
        self.viz_type_selector.addItem("Points and Lines")
        # Disable other types but keep them in the dropdown for future use
        self.viz_type_selector.setEnabled(False)  # Force "Points and Lines" only
        options_grid.addWidget(self.viz_type_selector, row, 0)
        
        options_group.setLayout(options_grid)
        options_layout.addWidget(options_group)
        
        # Shot information
        info_group = QGroupBox("Shot Information")
        info_layout = QVBoxLayout()
        
        self.shot_info_label = QLabel("Select a shot to see details.")
        self.shot_info_label.setWordWrap(True)
        self.shot_info_label.setStyleSheet("""
            background-color: white;
            padding: 10px;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
        """)
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
    
    def debug_print(self, message):
        """Print debug messages to console."""
        print(f"DEBUG [Visualization]: {message}", file=sys.stderr)
        sys.stderr.flush()

    def setup_3d_plot(self):
        """Set up the initial 3D plot with professional styling."""
        # Resilient check
        if not hasattr(self, 'ax') or self.ax is None:
            return
        
        if hasattr(self, '_is_closing') and self._is_closing:
            return
            
        self.ax.clear()
        
        # Set labels with professional formatting
        self.ax.set_xlabel('X', fontsize=12, fontweight='bold', color='#455A64')
        self.ax.set_ylabel('Y', fontsize=12, fontweight='bold', color='#455A64')
        self.ax.set_zlabel('Z', fontsize=12, fontweight='bold', color='#455A64')
        
        # Set a more extreme view angle to better show 3D depth
        # Try different elevation and azimuth for better 3D effect
        self.ax.view_init(elev=90, azim=90)  # This should give a better view of depth
        
        # Set aspect ratio to 'auto' to avoid distortion
        self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
        
        # DON'T invert axes - this may be causing confusion
        # Keep the natural orientation for now
        
        # Set title with professional formatting
        self.ax.set_title('3D Joint Visualization', fontsize=14, fontweight='bold', color='#263238')
        
        # Add grid with better styling
        self.ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        
        # Set figure face color for better integration with UI
        if hasattr(self, 'figure') and self.figure:
            self.figure.patch.set_facecolor('#FAFAFA')
        self.ax.set_facecolor('#F5F5F5')
        
        # Add subtle box for better 3D perception
        if hasattr(self.ax, 'xaxis') and hasattr(self.ax.xaxis, 'pane'):
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('#ECEFF1')
            self.ax.yaxis.pane.set_edgecolor('#ECEFF1')
            self.ax.zaxis.pane.set_edgecolor('#ECEFF1')
        
        # Improve axis tick formatting
        self.ax.tick_params(axis='x', colors='#546E7A', labelsize=10)
        self.ax.tick_params(axis='y', colors='#546E7A', labelsize=10)
        self.ax.tick_params(axis='z', colors='#546E7A', labelsize=10)
        
        # Draw canvas with safety checks
        try:
            if hasattr(self, 'figure') and self.figure and hasattr(self, 'canvas') and self.canvas:
                self.figure.tight_layout()
                self.canvas.draw()
        except Exception as e:
            print(f"Error drawing canvas: {e}")

    def rotate_skeleton(self, angle_degrees):
        """
        Set the skeleton rotation to the specified angle around the Y axis.
        This rotates the data itself rather than changing the view angle.
        
        Args:
            angle_degrees: Absolute angle to set (0=front, 90=left, 180=back, 270=right)
        """
        if not hasattr(self, 'selected_shots') or not self.selected_shots:
            return
        
        # Store the new rotation angle
        self.skeleton_rotation_angle = angle_degrees
        
        # Update the visualization with the new rotation angle
        self.update_visualization()

    def rotate_point_y(self, x, y, z, angle_degrees):
        """
        Rotate a point around the Y axis by the given angle.
        
        Args:
            x, y, z: Coordinates of the point (unscaled)
            angle_degrees: Angle to rotate by, in degrees
            
        Returns:
            Tuple of new (x, y, z) coordinates after rotation
        """
        import math
        
        angle_rad = math.radians(angle_degrees)
        new_x = x * math.cos(angle_rad) + z * math.sin(angle_rad)
        new_z = -x * math.sin(angle_rad) + z * math.cos(angle_rad)
        
        return new_x, y, new_z
    

    def reset_3d_view(self):
        """Reset both the view angle and the skeleton rotation."""
        if self._is_closing or not hasattr(self, 'ax') or not self.ax:
            return
        
        # Reset view angle to 90, 90
        self.ax.view_init(elev=90, azim=90)
        
        # Reset skeleton rotation to front view (0 degrees)
        self.skeleton_rotation_angle = 0
        
        # Update the visualization
        self.update_visualization()

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
            
            # Create list item with updated score reference
            item_text = f"Shot {i+1} - {timestamp} - Score: {shot['subjective_score']}"
            item = QListWidgetItem(item_text)
            
            # Store shot data as item data
            item.setData(Qt.ItemDataRole.UserRole, shot)
            
            self.shots_list.addItem(item)
    
    def on_shot_selection_changed(self):
        """Handle shot selection changes with immediate visualization."""
        selected_items = self.shots_list.selectedItems()
        
        if not selected_items:
            self.shot_info_label.setText("Select a shot to see details.")
            self.setup_3d_plot()  # Reset plot
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
            info_text += f"Score: {shot['subjective_score']}\n\n"
            
            # Add metrics
            metrics = shot['metrics']
            if 'follow_through_score' in metrics:
                info_text += f"Follow-through: {metrics['follow_through_score']:.2f}\n"
            
            if 'sway_velocity' in metrics:
                # Calculate average sway
                sway_values = [metrics['sway_velocity'].get(joint, 0) for joint in 
                            ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'NOSE']]
                if sway_values:
                    avg_sway = sum(sway_values) / len(sway_values)
                    info_text += f"Avg Sway: {avg_sway:.2f} mm/s\n"
            
            self.shot_info_label.setText(info_text)
        else:
            # Multiple shots selected
            self.shot_info_label.setText(f"{len(self.selected_shots)} shots selected.")
        
        # Always update visualization immediately when selection changes
        self.update_visualization()
        
        # Force "Points and Lines" visualization type
        self.viz_type_selector.setCurrentText("Points and Lines")
    
    def on_joint_visibility_changed(self):
        """Handle joint visibility checkbox changes."""
        # Update visibility settings
        for joint, checkbox in self.joint_checkboxes.items():
            self.joint_visibility[joint] = checkbox.isChecked()
        
        # Update visualization
        self.update_visualization()
    
    def update_visualization(self):
        """Update the 3D visualization based on selected shots."""
        if self._is_closing or not hasattr(self, 'ax') or not self.ax:
            return
            
        if not self.selected_shots:
            self.setup_3d_plot()  # Reset plot
            self.coord_label.setHtml("Select a shot to see coordinates")
            return
        
        # Clear the plot
        self.ax.clear()
        
        # Set labels with professional formatting
        self.ax.set_xlabel('X', fontsize=12, fontweight='bold', color='#455A64')
        self.ax.set_ylabel('Z', fontsize=12, fontweight='bold', color='#455A64')
        self.ax.set_zlabel('Y', fontsize=12, fontweight='bold', color='#455A64')
        
        # Set a more effective view angle for better 3D depth perception
        self.ax.view_init(elev=90, azim=90)  # This gives better depth visibility
        
        # Maintain proper proportions with equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio prevents distortion
        
        # Color map for multiple shots
        colors = ['#1976D2', '#D32F2F', '#388E3C', '#7B1FA2', '#FFA000', '#0097A7', '#5D4037']
        
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
        
        # Initialize HTML formatted coordinate text for better display
        coordinate_html = "<style>table {width:100%;} td {padding:1px 3px;}</style><table>"
        coordinate_html += "<tr><th colspan='4'>Coordinates (X, Y, Z)</th></tr>"
        
        # Process each selected shot
        for shot_idx, shot in enumerate(self.selected_shots):
            color = colors[shot_idx % len(colors)]
            
            # Extract joint positions from metrics
            joint_positions = {}
            if 'joint_positions' in shot['metrics'] and shot['metrics']['joint_positions']:
                joint_positions = shot['metrics']['joint_positions']
                print(f"Using stored joint positions for shot {shot['id']}")
            else:
                # Try to get it from other metrics if available
                if 'sway_velocity' in shot['metrics']:
                    # This is a fallback approach - we don't have real positions but can create placeholders
                    # for visualization purposes - in real app you'd want to store actual joint positions
                    sway_data = shot['metrics']['sway_velocity']
                    
                    # Create a basic layout for common joints
                    joint_positions = {
                        'NOSE': {'x': 320, 'y': 100, 'z': 0},
                        'LEFT_SHOULDER': {'x': 280, 'y': 150, 'z': 0},
                        'RIGHT_SHOULDER': {'x': 360, 'y': 150, 'z': 0},
                        'LEFT_ELBOW': {'x': 250, 'y': 200, 'z': 20},
                        'RIGHT_ELBOW': {'x': 390, 'y': 200, 'z': 20},
                        'LEFT_WRIST': {'x': 220, 'y': 250, 'z': 40},
                        'RIGHT_WRIST': {'x': 420, 'y': 250, 'z': 40},
                        'LEFT_HIP': {'x': 290, 'y': 300, 'z': 0},
                        'RIGHT_HIP': {'x': 350, 'y': 300, 'z': 0},
                        'LEFT_ANKLE': {'x': 280, 'y': 450, 'z': 0},
                        'RIGHT_ANKLE': {'x': 360, 'y': 450, 'z': 0}
                    }
                    
                    # Add variation based on sway data if available
                    import random
                    for joint in joint_positions:
                        if joint in sway_data:
                            sway = sway_data[joint]
                            random.seed(int(sway * 100) + shot['id'])  # Make it deterministic but different across shots
                            
                            offset_factor = min(sway / 10, 1.0)  # Normalize sway to reasonable range
                            
                            # Apply small offsets
                            joint_positions[joint]['x'] += random.uniform(-20, 20) * offset_factor
                            joint_positions[joint]['y'] += random.uniform(-20, 20) * offset_factor
                            joint_positions[joint]['z'] += random.uniform(-10, 10) * offset_factor
                    
                    print(f"Generated positions for shot {shot['id']}")
                else:
                    # No position data available
                    print(f"No position data available for shot {shot['id']}")
                    continue

            # In the update_visualization function, modify this part:
            rotated_joint_positions = {}
            for joint, pos in joint_positions.items():
                # Handle various data formats
                if isinstance(pos, list):
                    if not pos:
                        continue
                    pos = pos[0]
                
                if not all(key in pos for key in ['x', 'y', 'z']):
                    continue
                    
                # Get the original unscaled coordinates
                x, y, z = pos['x'], pos['y'], pos['z']
                
                # First rotate the unscaled coordinates
                new_x, new_y, new_z = self.rotate_point_y(x, y, z, self.skeleton_rotation_angle)
                
                # Store the rotated coordinates
                rotated_joint_positions[joint] = {'x': new_x, 'y': new_y, 'z': new_z}
            # Plot the skeleton with the rotated positions        
            # Only plot the skeleton once, with rotated positions
            try:
                self.plot_skeleton(rotated_joint_positions, connections, color, shot_idx)
                
                # For the first shot, build the coordinates table
                if shot_idx == 0:
                    # Filter to just show important joints to save space
                    important_joints = [
                        'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
                        'LEFT_ELBOW', 'RIGHT_ELBOW', 
                        'LEFT_WRIST', 'RIGHT_WRIST'
                    ]
                    
                    # Sort joints in a logical order
                    sorted_joints = [j for j in important_joints if j in rotated_joint_positions]
                    
                    # Add any other joints that might be in the data
                    for joint in sorted(rotated_joint_positions.keys()):
                        if joint not in sorted_joints:
                            sorted_joints.append(joint)
                    
                    # Build table rows for each joint
                    for joint in sorted_joints:
                        pos = rotated_joint_positions.get(joint)
                        if not pos:
                            continue
                        
                        # Handle list case
                        if isinstance(pos, list):
                            if not pos:
                                continue
                            pos = pos[0]
                        
                        # Check for required keys
                        if not all(key in pos for key in ['x', 'y', 'z']):
                            continue
                        
                        x, y, z = pos['x'], pos['y'], pos['z'] * 640
                        # Format with just one decimal place to save space
                        coordinate_html += f"<tr><td>{joint}:</td><td>{x:.1f}</td><td>{y:.1f}</td><td>{z:.1f}</td></tr>"
            except Exception as e:
                print(f"Error plotting shot {shot['id']}: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Close the HTML table
        coordinate_html += "</table>"
        
        # Set the HTML content instead of plain text
        self.coord_label.setHtml(coordinate_html)
        
        # Set plot title based on selected shots
        if len(self.selected_shots) == 1:
            self.ax.set_title(f'Shot #{self.selected_shots[0]["id"]} - Score: {self.selected_shots[0]["subjective_score"]}')
        else:
            self.ax.set_title(f'Comparing {len(self.selected_shots)} Shots')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        
        # Force redraw of the canvas
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Force GUI update to ensure coordinates are shown
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
    def _get_joint_group(self, joint_name):
        """
        Map individual joint names to their group categories for visualization filtering.

        Args:
            joint_name: Name of the joint from the tracking data

        Returns:
            Group name that corresponds to the visibility checkboxes
        """
        mapping = {
            'LEFT_ANKLE': 'ANKLES', 'RIGHT_ANKLE': 'ANKLES',
            'LEFT_HIP': 'HIPS', 'RIGHT_HIP': 'HIPS',
            'LEFT_SHOULDER': 'SHOULDERS', 'RIGHT_SHOULDER': 'SHOULDERS',
            'LEFT_ELBOW': 'ELBOWS', 'RIGHT_ELBOW': 'ELBOWS',
            'LEFT_WRIST': 'WRISTS', 'RIGHT_WRIST': 'WRISTS',
            'NOSE': 'NOSE'
        }
        return mapping.get(joint_name, joint_name)

    def plot_skeleton(self, joint_positions, connections, color, shot_idx):
        """
        Plot the skeleton with points and lines with improved visualization.
        
        Args:
            joint_positions: Dictionary of joint positions
            connections: List of joint connections
            color: Color for this skeleton (used when comparing multiple shots)
            shot_idx: Index of the shot
        """
        # Define colors for different body parts
        body_part_colors = {
            'SHOULDERS': '#1976D2',  # Blue
            'LEFT_SHOULDER': '#1976D2',
            'RIGHT_SHOULDER': '#1976D2',
            
            'ELBOWS': '#D32F2F',     # Red
            'LEFT_ELBOW': '#D32F2F',
            'RIGHT_ELBOW': '#D32F2F',
            
            'WRISTS': '#388E3C',     # Green
            'LEFT_WRIST': '#388E3C',
            'RIGHT_WRIST': '#388E3C',
            
            'NOSE': '#7B1FA2',       # Purple
            
            'HIPS': '#FFA000',       # Orange
            'LEFT_HIP': '#FFA000',
            'RIGHT_HIP': '#FFA000',
            
            'ANKLES': '#0097A7',     # Cyan
            'LEFT_ANKLE': '#0097A7',
            'RIGHT_ANKLE': '#0097A7'
        }
        
        # Determine if we're in comparison mode (multiple shots selected)
        comparison_mode = len(self.selected_shots) > 1
        
        # Mapping from joint names to abbreviations
        joint_labels = {
            'NOSE': 'NO',
            'LEFT_SHOULDER': 'LS',
            'RIGHT_SHOULDER': 'RS',
            'LEFT_ELBOW': 'LE',
            'RIGHT_ELBOW': 'RE',
            'LEFT_WRIST': 'LW',
            'RIGHT_WRIST': 'RW',
            'LEFT_HIP': 'LH',
            'RIGHT_HIP': 'RH',
            'LEFT_ANKLE': 'LA',
            'RIGHT_ANKLE': 'RA'
        }
        
        # Track min/max coordinates for proper scaling
        x_vals, y_vals, z_vals = [], [], []
        
        # Plot joints as points
        for joint_name, pos in joint_positions.items():
            # Skip if joint data is invalid
            if pos is None:
                continue
                
            joint_group = self._get_joint_group(joint_name)
            if joint_group not in self.joint_visibility or not self.joint_visibility[joint_group]:
                continue
            
            # In case we have multiple samples for the joint
            # In the plot_skeleton function, when you're getting the coordinates:
            if isinstance(pos, list):
                if not pos:  # Check if list is empty
                    continue
                pos = pos[0]

            if not all(key in pos for key in ['x', 'y', 'z']):
                print(f"Missing coordinate data for joint {joint_name}: {pos}")
                continue

            # Apply scaling to Z after rotation
            x, y, z = pos['x'], pos['y'], pos['z'] * 640

            # Add to coordinate lists for scaling
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            
            # Choose color based on mode:
            # In comparison mode - use the shot color to distinguish between skeletons
            # In single shot mode - use the body part colors for better visualization
            if comparison_mode:
                joint_color = color  # Use shot color in comparison mode
            else:
                joint_color = body_part_colors.get(joint_name, color)  # Use body part color in single shot mode
            
            # Plot the joint
            self.ax.scatter([x], [y], [z], color=joint_color, s=30, 
                            alpha=0.8, edgecolors='white', linewidths=1,
                            label=f"{joint_name}" if shot_idx == 0 else "")
            
            # Add text label for the joint
            label = joint_labels.get(joint_name, joint_name)
            self.ax.text(x, y, z, label, fontsize=8,
                        ha='center', va='center', color='black')
            
        # Plot connections as lines with matching colors
        for joint1, joint2 in connections:
            joint1_group = self._get_joint_group(joint1)
            joint2_group = self._get_joint_group(joint2)
            
            if (joint1_group not in self.joint_visibility or 
                joint2_group not in self.joint_visibility or
                not self.joint_visibility[joint1_group] or 
                not self.joint_visibility[joint2_group] or
                joint1 not in joint_positions or 
                joint2 not in joint_positions):
                continue
            
            pos1 = joint_positions[joint1]
            pos2 = joint_positions[joint2]
            
            # Handle multiple samples
            if isinstance(pos1, list):
                if not pos1:
                    continue
                pos1 = pos1[0]
            if isinstance(pos2, list):
                if not pos2:
                    continue
                pos2 = pos2[0]
            
            if not all(key in pos1 for key in ['x', 'y', 'z']) or not all(key in pos2 for key in ['x', 'y', 'z']):
                continue
            
            # Remap coordinates consistently
            x1, y1, z1 = pos1['x'], pos1['y'], pos1['z'] * 640
            x2, y2, z2 = pos2['x'], pos2['y'], pos2['z'] * 640
            
            # Choose color based on mode (same logic as for joints)
            if comparison_mode:
                connection_color = color  # Use shot color in comparison mode
            else:
                # In single shot mode, use body part colors
                # We could use either joint's color - here we're using joint1's
                connection_color = body_part_colors.get(joint1, color)
            
            self.ax.plot([x1, x2], [y1, y2], [z1, z2], color=connection_color, linewidth=1.5, alpha=0.7)
        
        # Axis limit adjustments...
        if x_vals and y_vals and z_vals:
            try:
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals) 
                z_min, z_max = min(z_vals), max(z_vals)
                
                x_range = max(x_max - x_min, 1)
                y_range = max(y_max - y_min, 1)
                z_range = max(z_max - z_min, 1)
                
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                z_center = (z_min + z_max) / 2
                
                max_range = max(x_range, y_range, z_range)
                padding = max_range * 0.6
                
                self.ax.set_xlim3d([x_center - padding, x_center + padding])
                self.ax.set_ylim3d([y_center - padding, y_center + padding])
                self.ax.set_zlim3d([z_center - padding, z_center + padding])
                
                self.ax.set_box_aspect([1.0, 1.0, 1.0])
            except Exception as e:
                print(f"Error setting axis limits: {e}")

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

    def set_session(self, session_id: int):
        """
        Set the current session and load shots.
        
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
        
        # If not found in selector, add it
        if session_id > 0:
            # Get session details
            self.cursor = self.data_storage.conn.cursor()
            self.cursor.execute("SELECT name, created_at FROM sessions WHERE id = ?", (session_id,))
            session = self.cursor.fetchone()
            
            if session:
                session_text = f"{session['name']} ({session['created_at'][:10]})"
                # Add to dropdown if not already there
                self.session_selector.addItem(session_text, session_id)
                self.session_selector.setCurrentIndex(self.session_selector.count() - 1)
            else:
                # If session not found, just set ID and load shots
                self.session_id = session_id
                self.load_session_shots()

    def _create_matplotlib_figure(self):
        """Create the matplotlib figure and canvas - separated for better control"""
        # Create 3D figure with improved styling and larger size
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        
        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        
        # Create 3D axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Initial plot setup
        self.setup_3d_plot()

    def safe_draw(self):
        """Safely draw the canvas with error handling and recreation if needed."""
        if hasattr(self, '_is_closing') and self._is_closing:
            return
            
        try:
            # Check and recreate canvas if needed
            recreated = self._ensure_canvas_valid()
            
            # Only proceed if canvas is now valid
            if hasattr(self, 'canvas') and self.canvas:
                if hasattr(self, 'figure') and self.figure:
                    self.figure.tight_layout()
                self.canvas.draw_idle()  # Using draw_idle() is safer than draw()
        except Exception as e:
            print(f"Error in safe_draw: {e}")
            traceback.print_exc()

    def closeEvent(self, event):
        """Handle the widget close event by cleaning up matplotlib resources."""
        self._is_closing = True
        # Explicitly clean up matplotlib resources
        if hasattr(self, 'figure') and self.figure:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
        super().closeEvent(event)
    
    def _ensure_canvas_valid(self):
        """Check if canvas is valid, recreate if needed."""
        if not hasattr(self, 'canvas') or self.canvas is None or not hasattr(self, 'figure') or self.figure is None:
            print("Canvas or figure not found, recreating")
            
            # Recreate figure and canvas
            self.figure = Figure(figsize=(10, 8), dpi=100)
            self.figure.patch.set_facecolor('#FAFAFA')
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.canvas.setMinimumHeight(400)
            
            # Recreate axes
            self.ax = self.figure.add_subplot(111, projection='3d')
            
            # Find and replace canvas in the layout
            if hasattr(self, 'canvas_parent_layout'):
                old_index = self.canvas_parent_layout.indexOf(self.old_canvas)
                if old_index >= 0:
                    self.canvas_parent_layout.removeWidget(self.old_canvas)
                    self.canvas_parent_layout.insertWidget(old_index, self.canvas, stretch=3)
                    self.old_canvas.deleteLater()
                    self.old_canvas = self.canvas
            
            # Setup the plot again
            self.setup_3d_plot()
            return True
        
        return False