from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QTabWidget, QMessageBox, QCheckBox,
    QGridLayout, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QBrush, QColor

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
        Initialize the dashboard widget with enhanced tabbed interface.
        
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
        """Initialize the user interface elements with tabbed interface."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Dashboard")
        title_label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1565C0;")
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
        self.session_stats_label.setStyleSheet("""
            background-color: #E3F2FD;
            padding: 5px 10px;
            border-radius: 4px;
            color: #1565C0;
            font-weight: bold;
        """)
        session_layout.addWidget(self.session_stats_label)
        
        main_layout.addLayout(session_layout)
        
        # Create tabbed interface for different dashboard views
        self.dashboard_tabs = QTabWidget()
        self.dashboard_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CFD8DC;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                min-width: 100px;
                padding: 8px 12px;
                margin-right: 2px;
            }
        """)
        
        # Create the three tab pages
        self.create_data_table_tab()
        self.create_graphs_tab()
        self.create_advanced_analytics_tab()
        
        main_layout.addWidget(self.dashboard_tabs)
        
        self.setLayout(main_layout)

    def create_data_table_tab(self):
        """Create the data table tab with export functionality."""
        data_tab = QWidget()
        layout = QVBoxLayout()
        
        # Top controls with export button
        controls_layout = QHBoxLayout()
        
        export_button = QPushButton("Export to CSV")
        export_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        export_button.clicked.connect(self.export_to_csv)
        controls_layout.addWidget(export_button)
        
        controls_layout.addStretch()
        
        # Search/filter controls could be added here
        
        layout.addLayout(controls_layout)
        
        # Shot data table with improved styling
        self.shots_table = QTableWidget()
        self.shots_table.setColumnCount(7)  # Added one column for export checkbox
        self.shots_table.setHorizontalHeaderLabels([
            "Select", "Shot #", "Timestamp", "Subjective Score", 
            "Follow-through", "Sway Velocity", "Postural Stability"
        ])
        
        # Set column widths
        self.shots_table.setColumnWidth(0, 60)  # Select checkbox
        
        # Set table styling
        self.shots_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #F5F8FA;
                gridline-color: #E1E8ED;
                selection-background-color: #BBDEFB;
            }
            QHeaderView::section {
                background-color: #1976D2;
                color: white;
                padding: 5px;
                border: 1px solid #1565C0;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        self.shots_table.setAlternatingRowColors(True)
        self.shots_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.shots_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Fixed width for checkbox
        
        layout.addWidget(self.shots_table)
        
        data_tab.setLayout(layout)
        self.dashboard_tabs.addTab(data_tab, "Data Table")

    def create_graphs_tab(self):
        """Create the graphs tab with multiple visualizations."""
        graphs_tab = QWidget()
        layout = QVBoxLayout()
        
        # Create splitter for arranging graphs
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top graphs section
        top_graphs = QWidget()
        top_layout = QHBoxLayout()
        
        # Score trend chart
        score_widget = QWidget()
        score_layout = QVBoxLayout()
        score_layout.addWidget(QLabel("Score Trends"))
        
        self.score_fig = Figure(figsize=(5, 3), dpi=100)
        self.score_fig.patch.set_facecolor('#FAFAFA')
        self.score_canvas = FigureCanvas(self.score_fig)
        score_layout.addWidget(self.score_canvas)
        
        score_widget.setLayout(score_layout)
        top_layout.addWidget(score_widget)
        
        # Stability metrics chart
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(QLabel("Joint Stability Metrics"))
        
        self.metrics_fig = Figure(figsize=(5, 3), dpi=100)
        self.metrics_fig.patch.set_facecolor('#FAFAFA')
        self.metrics_canvas = FigureCanvas(self.metrics_fig)
        metrics_layout.addWidget(self.metrics_canvas)
        
        metrics_widget.setLayout(metrics_layout)
        top_layout.addWidget(metrics_widget)
        
        top_graphs.setLayout(top_layout)
        splitter.addWidget(top_graphs)
        
        # Bottom graph - stability vs time correlation
        bottom_graph = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(QLabel("Stability vs. Time Correlation"))
        
        self.stability_time_fig = Figure(figsize=(10, 4), dpi=100)
        self.stability_time_fig.patch.set_facecolor('#FAFAFA')
        self.stability_time_canvas = FigureCanvas(self.stability_time_fig)
        bottom_layout.addWidget(self.stability_time_canvas)
        
        bottom_graph.setLayout(bottom_layout)
        splitter.addWidget(bottom_graph)
        
        # Set initial sizes
        splitter.setSizes([400, 300])
        
        layout.addWidget(splitter)
        
        graphs_tab.setLayout(layout)
        self.dashboard_tabs.addTab(graphs_tab, "Graphs")

    def create_advanced_analytics_tab(self):
        """Create the advanced analytics tab."""
        analytics_tab = QWidget()
        layout = QVBoxLayout()
        
        # Performance metrics header
        header = QLabel("Performance Analytics")
        header.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        header.setStyleSheet("color: #1565C0; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Create a grid layout for analytics cards
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        
        # Create analytics cards
        self.create_analytics_card(grid_layout, 0, 0, "Average Score", "0.0", "trophy")
        self.create_analytics_card(grid_layout, 0, 1, "Stability Trend", "Stable", "chart-line")
        self.create_analytics_card(grid_layout, 0, 2, "Sessions", "0", "calendar")
        self.create_analytics_card(grid_layout, 1, 0, "Total Shots", "0", "bullseye")
        self.create_analytics_card(grid_layout, 1, 1, "Best Session", "None", "star")
        self.create_analytics_card(grid_layout, 1, 2, "Follow-through Quality", "Good", "check")
        
        layout.addLayout(grid_layout)
        
        # Add correlation visualization
        correlation_header = QLabel("Performance Correlations")
        correlation_header.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        correlation_header.setStyleSheet("color: #1565C0; margin: 15px 0 10px 0;")
        layout.addWidget(correlation_header)
        
        # Correlation chart
        self.correlation_fig = Figure(figsize=(6, 4), dpi=100)
        self.correlation_fig.patch.set_facecolor('#FAFAFA')
        self.correlation_canvas = FigureCanvas(self.correlation_fig)
        layout.addWidget(self.correlation_canvas)
        
        # Initialize correlation chart
        self.setup_correlation_chart()
        
        analytics_tab.setLayout(layout)
        self.dashboard_tabs.addTab(analytics_tab, "Advanced Analytics")

    def create_analytics_card(self, parent_layout, row, col, title, value, icon_name):
        """Create an analytics card with a metric."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E1E8ED;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Card title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #657786; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Card value
        value_label = QLabel(value)
        value_label.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        value_label.setStyleSheet("color: #1565C0; margin: 5px 0;")
        layout.addWidget(value_label)
        
        # Save reference to update later
        # In create_analytics_card method of DashboardWidget
        setattr(self, f"{title.lower().replace(' ', '_').replace('-', '_')}_label", value_label)
        
        card.setLayout(layout)
        parent_layout.addWidget(card, row, col)
        
        return card

    def setup_correlation_chart(self):
        """Initialize the correlation chart with placeholder data."""
        ax = self.correlation_fig.add_subplot(111)
        
        # Placeholder data
        metrics = ['Score', 'Follow-through', 'Wrist Sway', 'Elbow Sway', 'Head Stability']
        correlations = [1.0, 0.85, -0.65, -0.55, 0.75]  # Positive and negative correlations
        
        # Create horizontal bar chart
        bars = ax.barh(metrics, correlations, color=['#2196F3' if c > 0 else '#F44336' for c in correlations])
        
        # Add labels and styling
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Correlation with Shot Score')
        ax.set_title('Performance Metric Correlations')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.05 if width > 0 else width - 0.05
            alignment = 'left' if width > 0 else 'right'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    ha=alignment, va='center', color='#212121')
        
        self.correlation_fig.tight_layout()
        self.correlation_canvas.draw()

    def update_shots_table(self, shots: List[Dict]):
        """
        Update the shots table with shot data and export checkboxes.
        
        Args:
            shots: List of shot dictionaries
        """
        self.shots_table.setRowCount(0)  # Clear table
        
        for i, shot in enumerate(shots):
            row_position = self.shots_table.rowCount()
            self.shots_table.insertRow(row_position)
            
            # Add checkbox for export
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Default to checked
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.shots_table.setCellWidget(row_position, 0, checkbox_widget)
            
            # Shot number
            self.shots_table.setItem(row_position, 1, QTableWidgetItem(str(i + 1)))
            
            # Timestamp (just show time part)
            timestamp = shot['timestamp'].split('T')[1][:8]  # HH:MM:SS
            self.shots_table.setItem(row_position, 2, QTableWidgetItem(timestamp))
            
            # Subjective score
            score_item = QTableWidgetItem(str(shot['subjective_score']))
            # Color code based on score
            if shot['subjective_score'] >= 8:
                score_item.setForeground(QBrush(QColor('#388E3C')))  # Green for high scores
            elif shot['subjective_score'] <= 4:
                score_item.setForeground(QBrush(QColor('#D32F2F')))  # Red for low scores
            self.shots_table.setItem(row_position, 3, score_item)
            
            # Follow-through score
            follow_through = shot['metrics'].get('follow_through_score', 0)
            follow_through_item = QTableWidgetItem(f"{follow_through:.2f}")
            self.shots_table.setItem(row_position, 4, follow_through_item)
            
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
                
            self.shots_table.setItem(row_position, 5, sway_item)
            
            # Postural stability (average of DevX and DevY)
            dev_x = shot['metrics'].get('dev_x', {}).get('UPPER_BODY', 0)
            dev_y = shot['metrics'].get('dev_y', {}).get('UPPER_BODY', 0)
            
            stability = (dev_x + dev_y) / 2
            stability_item = QTableWidgetItem(f"{stability:.2f} px")
            self.shots_table.setItem(row_position, 6, stability_item)

    def update_score_chart(self, shots: List[Dict]):
        """
        Update the score trend chart with professional styling.
        
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
        
        # Plot data with improved styling
        ax.plot(shot_numbers, subjective_scores, 'b-', marker='o', label='Subjective Score',
                color='#1976D2', linewidth=2, markersize=8, alpha=0.8)
        
        ax.plot(shot_numbers, follow_through_scores, 'g--', marker='s', label='Follow-through (×10)',
                color='#43A047', linewidth=2, markersize=6, alpha=0.7)
        
        # Set labels and title with professional formatting
        ax.set_xlabel('Shot Number', fontsize=10, fontweight='bold', color='#455A64')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold', color='#455A64')
        ax.set_title('Score Trends', fontsize=12, fontweight='bold', color='#1565C0')
        
        # Set y-axis limits
        ax.set_ylim(0, 11)
        
        # Add grid and legend with improved styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        legend = ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8)
        legend.get_frame().set_edgecolor('#CFD8DC')
        
        # Set background colors
        ax.set_facecolor('#F5F8FA')
        
        # Add moving average trend line if enough data points
        if len(subjective_scores) >= 3:
            window_size = min(3, len(subjective_scores))
            moving_avg = np.convolve(subjective_scores, np.ones(window_size)/window_size, mode='valid')
            
            # Plot moving average after padding for proper alignment
            padding = [None] * (len(subjective_scores) - len(moving_avg))
            ax.plot(shot_numbers, padding + list(moving_avg), 'r-', label='Trend',
                color='#FF7043', linewidth=2, alpha=0.7)
            
            # Update legend
            ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8)
        
        # Draw canvas
        self.score_fig.tight_layout()
        self.score_canvas.draw()

    def update_metrics_chart(self, shots: List[Dict]):
        """
        Update the stability metrics chart with professional styling.
        
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
        
        # Plot data with improved styling
        ax.plot(shot_numbers, wrist_sway, marker='o', label='Wrist Sway',
                color='#E53935', linewidth=2, markersize=6)
        
        ax.plot(shot_numbers, elbow_sway, marker='s', label='Elbow Sway',
                color='#43A047', linewidth=2, markersize=6)
        
        ax.plot(shot_numbers, shoulder_sway, marker='^', label='Shoulder Sway',
                color='#1E88E5', linewidth=2, markersize=6)
        
        # Set labels and title with professional formatting
        ax.set_xlabel('Shot Number', fontsize=10, fontweight='bold', color='#455A64')
        ax.set_ylabel('Sway Velocity (mm/s)', fontsize=10, fontweight='bold', color='#455A64')
        ax.set_title('Joint Stability Metrics', fontsize=12, fontweight='bold', color='#1565C0')
        
        # Add grid and legend with improved styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        legend = ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8)
        legend.get_frame().set_edgecolor('#CFD8DC')
        
        # Set background colors
        ax.set_facecolor('#F5F8FA')
        
        # Draw canvas
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()

    def update_stability_time_chart(self, shots: List[Dict]):
        """
        Create a new stability vs time correlation chart.
        
        Args:
            shots: List of shot dictionaries
        """
        # Clear figure
        self.stability_time_fig.clear()
        
        if not shots or len(shots) < 2:
            self.stability_time_canvas.draw()
            return
        
        # Create subplot
        ax = self.stability_time_fig.add_subplot(111)
        
        # Parse timestamps
        import datetime
        timestamps = []
        for shot in shots:
            try:
                # Parse ISO format timestamp
                dt = datetime.datetime.fromisoformat(shot['timestamp'])
                timestamps.append(dt)
            except (ValueError, TypeError):
                # Fallback to sequential numbers if timestamp parsing fails
                timestamps.append(None)
        
        # If timestamp parsing failed, use shot numbers
        if None in timestamps:
            time_data = list(range(1, len(shots) + 1))
            time_label = 'Shot Number'
        else:
            # Convert to matplotlib dates
            import matplotlib.dates as mdates
            time_data = mdates.date2num(timestamps)
            time_label = 'Time'
            
            # Format x-axis as times
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Calculate overall stability scores
        stability_scores = []
        for shot in shots:
            # Get metrics
            metrics = shot['metrics']
            
            # Calculate average sway (lower is better)
            sway = 0
            count = 0
            if 'sway_velocity' in metrics:
                for value in metrics['sway_velocity'].values():
                    sway += value
                    count += 1
                if count > 0:
                    sway /= count
            
            # Get follow-through score (higher is better)
            follow_through = metrics.get('follow_through_score', 0.5)
            
            # Calculate overall stability (0-100, higher is better)
            # Convert sway to a 0-1 scale (0 = high sway, 1 = low sway)
            sway_factor = max(0, 1 - (sway / 20))  # Assume max sway of 20 mm/s
            
            # Weighted combination
            stability = (0.7 * follow_through + 0.3 * sway_factor) * 100
            stability_scores.append(stability)
        
        # Plot stability vs time with improved styling
        scatter = ax.scatter(
            time_data, 
            stability_scores, 
            c=stability_scores,  # Color by stability
            cmap='viridis',      # Professional color map
            s=80,                # Marker size
            alpha=0.8,           # Transparency
            edgecolor='white'    # White outline
        )
        
        # Add a best fit line
        import numpy as np
        from scipy import stats
        
        # Use shot indices for regression if timestamps failed
        x_for_reg = list(range(len(time_data))) if None in timestamps else time_data
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_for_reg, stability_scores)
        
        if None not in timestamps:
            # Format dates correctly for line plot
            reg_x = np.array([min(time_data), max(time_data)])
            reg_y = intercept + slope * np.array([0, len(time_data) - 1])
        else:
            reg_x = np.array([1, len(time_data)])
            reg_y = intercept + slope * np.array([0, len(time_data) - 1])
        
        ax.plot(reg_x, reg_y, 'r--', linewidth=2, alpha=0.7, label=f'Trend (r={r_value:.2f})')
        
        # Add a colorbar to show stability scale
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Stability Score', fontsize=10, fontweight='bold')
        
        # Set labels and title with professional formatting
        ax.set_xlabel(time_label, fontsize=10, fontweight='bold', color='#455A64')
        ax.set_ylabel('Stability Score (0-100)', fontsize=10, fontweight='bold', color='#455A64')
        ax.set_title('Stability vs. Time Analysis', fontsize=12, fontweight='bold', color='#1565C0')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        ax.legend(loc='lower right')
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Set background colors
        ax.set_facecolor('#F5F8FA')
        
        # Rotate x-axis labels if using timestamps
        if None not in timestamps:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Draw canvas
        self.stability_time_fig.tight_layout()
        self.stability_time_canvas.draw()

    def refresh_data(self):
        """Refresh all dashboard data for the current session and update all tabs."""
        if not self.current_session:
            return
        
        try:
            # Get session statistics
            stats = self.data_storage.get_session_stats(self.current_session)
            
            # Update stats label
            stats_text = f"Shots: {stats.get('shot_count', 0)} | "
            stats_text += f"Avg Score: {stats.get('avg_subjective_score', 0):.1f} | "
            stats_text += f"Best: {stats.get('max_subjective_score', 0)} | "
            stats_text += f"Worst: {stats.get('min_subjective_score', 0)}"
            
            self.session_stats_label.setText(stats_text)
            
            # Get shot data
            shots = self.data_storage.get_shots(self.current_session)
            
            # Update table tab
            self.update_shots_table(shots)
            
            # Update graphs tab
            self.update_score_chart(shots)
            self.update_metrics_chart(shots)
            self.update_stability_time_chart(shots)
            
            # Update analytics tab metrics
            self.update_analytics_metrics(shots, stats)
            
        except Exception as e:
            import traceback
            print(f"Error refreshing dashboard data: {e}")
            print(traceback.format_exc())
            self.session_stats_label.setText("Error loading data")

    def update_analytics_metrics(self, shots: List[Dict], stats: Dict):
        """
        Update the analytics metrics cards and correlation chart.
        
        Args:
            shots: List of shot dictionaries
            stats: Session statistics dictionary
        """
        # Update the metrics cards
        if hasattr(self, 'average_score_label'):
            self.average_score_label.setText(f"{stats.get('avg_subjective_score', 0):.1f}")
        
        if hasattr(self, 'total_shots_label'):
            self.total_shots_label.setText(f"{stats.get('shot_count', 0)}")
        
        # Determine stability trend
        if len(shots) >= 3:
            # Get stability scores for last 3 shots
            recent_stability = []
            for shot in shots[-3:]:
                follow_through = shot['metrics'].get('follow_through_score', 0.5)
                recent_stability.append(follow_through)
            
            # Calculate trend
            if len(recent_stability) == 3:
                if recent_stability[2] > recent_stability[0]:
                    trend = "Improving"
                    trend_color = "#388E3C"  # Green
                elif recent_stability[2] < recent_stability[0]:
                    trend = "Declining"
                    trend_color = "#D32F2F"  # Red
                else:
                    trend = "Stable"
                    trend_color = "#1976D2"  # Blue
                
                if hasattr(self, 'stability_trend_label'):
                    self.stability_trend_label.setText(trend)
                    self.stability_trend_label.setStyleSheet(f"color: {trend_color}; font-size: 24px; font-weight: bold;")
        
        # Get best session info from database
        if self.user_id and hasattr(self, 'best_session_label'):
            try:
                self.cursor = self.data_storage.conn.cursor()
                self.cursor.execute("""
                    SELECT s.name, AVG(sh.subjective_score) as avg_score
                    FROM sessions s
                    JOIN shots sh ON s.id = sh.session_id
                    WHERE s.user_id = ?
                    GROUP BY s.id
                    ORDER BY avg_score DESC
                    LIMIT 1
                """, (self.user_id,))
                best_session = self.cursor.fetchone()
                
                if best_session:
                    self.best_session_label.setText(f"{best_session['name']}")
            except Exception as e:
                print(f"Error getting best session: {e}")
        
        # Calculate average follow-through quality
        if shots and hasattr(self, 'follow-through_quality_label'):
            follow_through_scores = [shot['metrics'].get('follow_through_score', 0) for shot in shots]
            avg_follow_through = sum(follow_through_scores) / len(follow_through_scores)
            
            if avg_follow_through > 0.7:
                quality = "Excellent"
                quality_color = "#388E3C"  # Green
            elif avg_follow_through > 0.4:
                quality = "Good"
                quality_color = "#1976D2"  # Blue
            else:
                quality = "Needs Work"
                quality_color = "#D32F2F"  # Red
            
            self.follow_through_quality_label.setText(quality)
            self.follow_through_quality_label.setStyleSheet(f"color: {quality_color}; font-size: 24px; font-weight: bold;")

    def export_to_csv(self):
        """Export selected shots to CSV file."""
        if not self.current_session or self.shots_table.rowCount() == 0:
            QMessageBox.warning(self, "Export Error", "No data to export.")
            return
        
        # Get export filename
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "shooting_session_data.csv", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Collect data to export
            export_data = []
            headers = ["Shot #", "Timestamp", "Subjective Score", "Follow-through", 
                    "Sway Velocity", "Postural Stability"]
            
            for row in range(self.shots_table.rowCount()):
                # Check if this row is selected for export
                checkbox_widget = self.shots_table.cellWidget(row, 0)
                if checkbox_widget and checkbox_widget.findChild(QCheckBox).isChecked():
                    row_data = []
                    for col in range(1, self.shots_table.columnCount()):
                        item = self.shots_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    
                    export_data.append(row_data)
            
            # Write to CSV
            import csv
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(export_data)
            
            QMessageBox.information(self, "Export Successful", 
                                f"Data exported successfully to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")

    def on_session_changed(self, index: int):
        """
        Handle session selection change with improved error handling.
        
        Args:
            index: Index of the selected session in the dropdown
        """
        if index <= 0:  # "Select a session..." item
            self.current_session = None
            self.session_stats_label.setText("No session selected")
            self.clear_data()
            return
        
        try:
            # Get session ID from combobox data
            session_id = self.session_selector.itemData(index)
            
            if session_id > 0:
                self.current_session = session_id
                self.refresh_data()
        except Exception as e:
            print(f"Error in session change: {e}")
            self.session_stats_label.setText("Error loading session")
            self.clear_data()

    def clear_data(self):
        """Clear all data displays when no session is selected."""
        # Clear table
        if hasattr(self, 'shots_table'):
            self.shots_table.setRowCount(0)
        
        # Clear charts
        if hasattr(self, 'score_fig'):
            self.score_fig.clear()
            self.score_canvas.draw()
        
        if hasattr(self, 'metrics_fig'):
            self.metrics_fig.clear()
            self.metrics_canvas.draw()
            
        if hasattr(self, 'stability_time_fig'):
            self.stability_time_fig.clear()
            self.stability_time_canvas.draw()
        
        # Reset analytics metrics
        for attr_name in dir(self):
            if attr_name.endswith('_label') and 'stats' not in attr_name:
                label = getattr(self, attr_name)
                if isinstance(label, QLabel) and attr_name != 'session_stats_label':
                    if 'score' in attr_name:
                        label.setText("0.0")
                    elif 'shots' in attr_name:
                        label.setText("0")
                    else:
                        label.setText("N/A")
    def set_user(self, user_id: int):
        """
        Set the current user and load their sessions.
        
        Args:
            user_id: ID of the current user
        """
        self.user_id = user_id
        self.refresh_sessions()
        
        # Reset any user-specific data
        self.current_session = None
        self.clear_data()
        
        # Update analytics if needed
        if hasattr(self, 'correlation_fig'):
            self.setup_correlation_chart()

    def refresh_sessions(self):
        """Refresh the sessions dropdown with user's sessions."""
        if not self.user_id:
            return
        
        try:
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
                
        except Exception as e:
            print(f"Error refreshing sessions: {e}")
            # Ensure placeholder is added even if error occurs
            if self.session_selector.count() == 0:
                self.session_selector.addItem("Select a session...", -1)
    
    def set_session(self, session_id: int):
        """
        Set the current session and load data.
        
        Args:
            session_id: ID of the session
        """
        if not session_id or session_id <= 0:
            return
            
        # Update selector to match
        for i in range(self.session_selector.count()):
            if self.session_selector.itemData(i) == session_id:
                self.session_selector.setCurrentIndex(i)
                return
        
        # If not found in selector, add it
        if session_id > 0:
            try:
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
                    # If session not found, just set ID and refresh
                    self.current_session = session_id
                    self.refresh_data()
            except Exception as e:
                print(f"Error setting session in dashboard: {e}")
                # Direct method as fallback
                self.current_session = session_id
                self.refresh_data()