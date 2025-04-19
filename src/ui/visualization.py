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
        
        # Create 3D figure with improved styling
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Initial plot setup
        self.setup_3d_plot()
        
        viz_layout.addWidget(self.canvas)
        
        # Add coordinate display
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Coordinates:"))
        self.coord_label = QLabel("Select a shot to see coordinates")
        self.coord_label.setStyleSheet("""
            background-color: #E3F2FD;
            padding: 5px;
            border-radius: 3px;
            font-family: monospace;
        """)
        coord_layout.addWidget(self.coord_label)
        viz_layout.addLayout(coord_layout)
        
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
    
    def setup_3d_plot(self):
        """Set up the initial 3D plot with professional styling."""
        self.ax.clear()
        
        # Set labels with professional formatting
        self.ax.set_xlabel('X', fontsize=12, fontweight='bold', color='#455A64')
        self.ax.set_ylabel('Z', fontsize=12, fontweight='bold', color='#455A64')  # In MediaPipe, Z is depth
        self.ax.set_zlabel('Y', fontsize=12, fontweight='bold', color='#455A64')  # In PyQt, Y is inverted (downward)
        
        # Set initial view
        self.ax.view_init(elev=20, azim=-60)
        
        # Set axis limits (will be adjusted based on data)
        self.ax.set_xlim3d([0, 640])
        self.ax.set_ylim3d([-100, 100])
        self.ax.set_zlim3d([0, 480])
        
        # Set title with professional formatting
        self.ax.set_title('3D Joint Visualization', fontsize=14, fontweight='bold', color='#263238')
        
        # Add grid with better styling
        self.ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        
        # Set figure face color for better integration with UI
        self.figure.patch.set_facecolor('#FAFAFA')
        self.ax.set_facecolor('#F5F5F5')
        
        # Add subtle box for better 3D perception
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
            info_text += f"Subjective Score: {shot['subjective_score']}\n\n"
            
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
        if not self.selected_shots:
            self.setup_3d_plot()  # Reset plot
            self.coord_label.setText("Select a shot to see coordinates")
            return
        
        # Clear the plot
        self.ax.clear()
        
        # Always use "Points and Lines" visualization
        viz_type = "Points and Lines"
        
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
        
        # Collect all coordinate data for display
        coordinate_text = ""
        
        # Process each selected shot
        for shot_idx, shot in enumerate(self.selected_shots):
            color = colors[shot_idx % len(colors)]
            
            # Extract joint positions from metrics
            if 'joint_positions' not in shot['metrics']:
                # Try to get it from other metrics if available
                joint_positions = {}
                
                # Check if joints info is available in another format
                if 'sway_velocity' in shot['metrics']:
                    # This is a fallback approach - we don't have real positions but can create placeholders
                    # for visualization purposes - in real app you'd want to store actual joint positions
                    placeholder_positions = {}
                    
                    # Use sway velocities to create a suggestive layout
                    sway_data = shot['metrics']['sway_velocity']
                    
                    # Create a basic layout for common joints
                    base_positions = {
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
                    for joint in base_positions:
                        if joint in sway_data:
                            # Small random offsets based on sway
                            import random
                            sway = sway_data[joint]
                            random.seed(sway + shot['id'])  # Make it deterministic but different across shots
                            
                            offset_factor = min(sway / 10, 1.0)  # Normalize sway to reasonable range
                            
                            # Apply small offsets
                            base_positions[joint]['x'] += random.uniform(-20, 20) * offset_factor
                            base_positions[joint]['y'] += random.uniform(-20, 20) * offset_factor
                            base_positions[joint]['z'] += random.uniform(-10, 10) * offset_factor
                    
                    joint_positions = base_positions
                else:
                    # No position data available
                    continue
            else:
                joint_positions = shot['metrics']['joint_positions']
            
            # Plot the skeleton
            self.plot_skeleton(joint_positions, connections, color, shot_idx)
            
            # Collect coordinate data for display
            if shot_idx == 0:  # Show coordinates for first selected shot
                coordinate_text = "Coordinates (X, Y, Z):\n"
                for joint in sorted(joint_positions.keys()):
                    pos = joint_positions[joint]
                    
                    # Handle if pos is a list
                    if isinstance(pos, list):
                        pos = pos[0]
                    
                    x, y, z = pos['x'], pos['z'], pos['y']  # Note the mapping
                    coordinate_text += f"{joint}: ({x:.1f}, {y:.1f}, {z:.1f})\n"
        
        # Update coordinate display
        if coordinate_text:
            self.coord_label.setText(coordinate_text)
        else:
            self.coord_label.setText("No coordinate data available")
        
        # Set plot properties
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')
        
        if len(self.selected_shots) == 1:
            self.ax.set_title(f'Shot #{self.selected_shots[0]["id"]} - Score: {self.selected_shots[0]["subjective_score"]}')
        else:
            self.ax.set_title(f'Comparing {len(self.selected_shots)} Shots')
        
        self.ax.grid(True)
        
        # Draw the updated plot
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_skeleton(self, joint_positions, connections, color, shot_idx):
        """
        Plot the skeleton with points and lines with improved visualization.
        
        Args:
            joint_positions: Dictionary of joint positions
            connections: List of joint connections
            color: Color for this skeleton
            shot_idx: Index of the shot
        """
        # Track min/max coordinates for proper scaling
        x_vals, y_vals, z_vals = [], [], []
        
        # Plot joints as points with better styling
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
            
            # Plot the joint with improved styling
            self.ax.scatter([x], [y], [z], color=color, s=70, 
                            alpha=0.8, edgecolors='white', linewidths=1,
                            label=f"{joint} (Shot {shot_idx+1})" if shot_idx == 0 else "")
        
        # Plot connections as lines with improved styling
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
            
            # Draw line with improved styling
            self.ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=2, alpha=0.7)
        
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
        
        # Add a legend if this is the first shot
        if shot_idx == 0:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                self.ax.legend(by_label.values(), by_label.keys(), 
                            loc='upper right', fontsize=8)
    
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
        Plot heatmap visualization of joint stability with professional styling.
        
        Args:
            joint_positions: Dictionary of joint positions
            shot_idx: Index of the shot
        """
        # Get sway velocities if available
        sway_velocities = self.selected_shots[shot_idx]['metrics'].get('sway_velocity', {})
        
        # Plot joints as points with color based on sway and improved styling
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
            
            # Professional color mapping using a proper color scale: green (stable) to red (unstable)
            normalized_sway = min(1.0, sway / 20.0)  # Cap at 20 mm/s
            
            # Better color interpolation
            if normalized_sway < 0.5:
                # Green to yellow
                r = 2 * normalized_sway
                g = 1.0
                b = 0.0
            else:
                # Yellow to red
                r = 1.0
                g = 2 * (1 - normalized_sway)
                b = 0.0
            
            color = [r, g, b]
            
            # Plot the joint with improved styling
            scatter = self.ax.scatter([x], [y], [z], color=color, s=120, alpha=0.8,
                                    edgecolors='white', linewidths=1,
                                    label=f"{joint} (Shot {shot_idx+1})" if shot_idx == 0 else "")
            
            # Add text label with sway value for better understanding
            self.ax.text(x, y, z, f"{sway:.1f}", color='white', fontsize=8, 
                        horizontalalignment='center', verticalalignment='center',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        
        # Add a colorbar to explain the heatmap
        if shot_idx == 0:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            # Create a custom colormap
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'StabilityMap', [(0, (0, 1, 0)), (0.5, (1, 1, 0)), (1, (1, 0, 0))]
            )
            
            # Add a colorbar
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
            cbar.set_label('Sway Velocity (mm/s)', fontsize=10, fontweight='bold')
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0 (Stable)', '10', '20+ (Unstable)'])
            
            # Update layout to accommodate colorbar
            self.figure.tight_layout()
    
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