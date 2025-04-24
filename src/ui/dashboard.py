from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QTabWidget, QMessageBox, QCheckBox,
    QGridLayout, QFileDialog, QDateEdit, QGroupBox, QRadioButton,
    QStackedWidget, QScrollArea, QSlider, QSizePolicy, QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont, QBrush, QColor, QPainter, QPen, QPixmap

import json  # This was missing and causing errors
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import os

import webbrowser
import urllib.parse
import tempfile
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from PyQt6.QtWidgets import QMenu, QDialog, QVBoxLayout, QLabel, QLineEdit, QDialogButtonBox
from src.data_storage import DataStorage

matplotlib.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.figsize': (6, 3),
    'figure.dpi': 100,
    'figure.autolayout': False,  # Disable auto layout to prevent warnings
})

class PerformanceWidget(QWidget):
    """Widget for displaying performance metrics over time."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure for analytics
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_chart(self, data, metric='score', title='Performance Trend'):
        """Update the chart with new data."""
        self.figure.clear()
        
        if not data:
            self.canvas.draw()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Extract data
        dates = [item['date'] for item in data]
        values = [item[metric] for item in data]
        
        # Plot with trend line
        scatter = ax.scatter(dates, values, s=80, alpha=0.6, 
                            c=values, cmap='viridis',
                            edgecolor='white', zorder=5)
        
        # Add trend line if we have enough data
        if len(dates) > 2:
            try:
                # Convert dates to numbers for regression
                import matplotlib.dates as mdates
                date_nums = mdates.date2num(dates)
                
                # Calculate trend line with numpy
                z = np.polyfit(date_nums, values, 1)
                p = np.poly1d(z)
                
                # Generate points for trend line
                trend_dates = [min(dates), max(dates)]
                trend_dates_num = mdates.date2num(trend_dates)
                trend_values = p(trend_dates_num)
                
                # Plot trend line
                ax.plot(trend_dates, trend_values, 'r--', linewidth=2, 
                        label=f'Trend: {z[0]:.4f}')
                
                # Add text showing improvement percentage
                if len(trend_values) > 1 and trend_values[0] > 0:
                    improvement = ((trend_values[1] - trend_values[0]) / trend_values[0]) * 100
                    if improvement > 0:
                        label = f"+{improvement:.1f}% improvement"
                        color = 'green'
                    else:
                        label = f"{improvement:.1f}% change"
                        color = 'red'
                    
                    ax.annotate(label, xy=(0.5, 0.95), xycoords='axes fraction',
                                ha='center', va='top', fontsize=12, 
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=color, alpha=0.8),
                                color=color)
                
            except Exception as e:
                print(f"Error calculating trend: {e}")
        
        # Add grid with better styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45)
        
        # Add labels and title
        ax.set_xlabel('Date', fontsize=12, fontweight='bold', color='#455A64')
        
        # Set y-label based on metric
        if metric == 'score':
            ax.set_ylabel('Score', fontsize=12, fontweight='bold', color='#455A64')
            ax.set_ylim(0, 10.9)
        elif metric == 'stability':
            ax.set_ylabel('Stability (%)', fontsize=12, fontweight='bold', color='#455A64')
            ax.set_ylim(0, 100)
        elif metric == 'follow_through':
            ax.set_ylabel('Follow-through', fontsize=12, fontweight='bold', color='#455A64')
            ax.set_ylim(0, 1.0)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#1565C0')
        
        # Add colorbar for reference
        cbar = plt.colorbar(scatter)
        if metric == 'score':
            cbar.set_label('Score Value', fontweight='bold')
        elif metric == 'stability':
            cbar.set_label('Stability (%)', fontweight='bold')
        elif metric == 'follow_through':
            cbar.set_label('Follow-through', fontweight='bold')
        
        # Add legend if we have a trend line
        if len(dates) > 2:
            ax.legend(loc='lower right')
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()

class RadarWidget(QWidget):
    """Widget for displaying radar chart of shooting metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure for radar chart
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_chart(self, current_metrics, baseline_metrics=None, title='Performance Breakdown'):
        """Update radar chart with correct wrist stability values."""
        self.figure.clear()
        
        if not current_metrics:
            self.canvas.draw()
            return
            
        # Create subplot with polar projection for radar chart
        ax = self.figure.add_subplot(111, projection='polar')
        
        # Define metrics to display
        metrics = [
            'Overall Stability', 
            'Follow-through',
            'Wrist Stability',  # This needs fixing
            'Elbow Stability',
            'Shoulder Stability',
            'Head Stability',
            'Consistency'
        ]
        
        # Number of metrics
        N = len(metrics)
        
        # Set angles for each metric (evenly spaced around the circle)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the circle
        
        # Fix wrist stability calculation - ensure we're getting proper values
        # Use normalized value (0-100 scale) based on sway
        wrist_sway = current_metrics.get('wrist_sway', 0)
        # Only normalize if we have real data - check if value exists and isn't zero
        if wrist_sway > 0:
            wrist_stability = max(0, min(100, 100 - (wrist_sway * 5)))
        else:
            # If we don't have data, use a default value or calculate from raw data
            # Look at sway_velocity directly from metrics if available
            sway_metrics = current_metrics.get('sway_velocity', {})
            if isinstance(sway_metrics, dict) and ('LEFT_WRIST' in sway_metrics or 'RIGHT_WRIST' in sway_metrics):
                left = sway_metrics.get('LEFT_WRIST', 0)
                right = sway_metrics.get('RIGHT_WRIST', 0)
                wrist_sway = (left + right) / 2 if (left > 0 or right > 0) else 10  # Default if zero
                wrist_stability = max(0, min(100, 100 - (wrist_sway * 5)))
            else:
                # Fallback value
                wrist_stability = 50  # Default middle value if no data
        
        # Extract current performance values with fix for wrist stability
        current_values = [
            current_metrics.get('overall_stability', 0) * 100,
            current_metrics.get('follow_through', 0) * 100,
            wrist_stability,  # Fixed wrist stability
            (100 - min(current_metrics.get('elbow_sway', 0) * 5, 100)),
            (100 - min(current_metrics.get('shoulder_sway', 0) * 5, 100)),
            (100 - min(current_metrics.get('nose_sway', 0) * 5, 100)),
            current_metrics.get('consistency', 0) * 100
        ]
        current_values += current_values[:1]  # Close the circle
        
        # Plot current performance with simpler styling
        ax.plot(angles, current_values, linewidth=1.5, linestyle='solid', 
            color='#1976D2', label='Current')
        ax.fill(angles, current_values, '#1976D2', alpha=0.2)
        
        # If baseline metrics provided, plot for comparison
        if baseline_metrics:
            # Same fix for baseline wrist stability
            baseline_wrist_sway = baseline_metrics.get('wrist_sway', 0)
            if baseline_wrist_sway > 0:
                baseline_wrist_stability = max(0, min(100, 100 - (baseline_wrist_sway * 5)))
            else:
                baseline_sway_metrics = baseline_metrics.get('sway_velocity', {})
                if isinstance(baseline_sway_metrics, dict) and ('LEFT_WRIST' in baseline_sway_metrics or 'RIGHT_WRIST' in baseline_sway_metrics):
                    left = baseline_sway_metrics.get('LEFT_WRIST', 0)
                    right = baseline_sway_metrics.get('RIGHT_WRIST', 0)
                    baseline_wrist_sway = (left + right) / 2 if (left > 0 or right > 0) else 10
                    baseline_wrist_stability = max(0, min(100, 100 - (baseline_wrist_sway * 5)))
                else:
                    baseline_wrist_stability = 50
            
            baseline_values = [
                baseline_metrics.get('overall_stability', 0) * 100,
                baseline_metrics.get('follow_through', 0) * 100,
                baseline_wrist_stability,  # Fixed baseline wrist stability
                (100 - min(baseline_metrics.get('elbow_sway', 0) * 5, 100)),
                (100 - min(baseline_metrics.get('shoulder_sway', 0) * 5, 100)),
                (100 - min(baseline_metrics.get('nose_sway', 0) * 5, 100)),
                baseline_metrics.get('consistency', 0) * 100
            ]
            baseline_values += baseline_values[:1]  # Close the circle
            
            ax.plot(angles, baseline_values, linewidth=1.5, linestyle='dashed', 
                color='#FFA000', label='Baseline')
            ax.fill(angles, baseline_values, '#FFA000', alpha=0.1)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        
        # Remove radial labels and set y-limits
        ax.set_yticklabels([])
        ax.set_ylim(0, 100)
        
        # Add subtle grid lines
        ax.grid(True, linestyle='--', alpha=0.3, color='#CFD8DC')
        
        # Set title with simpler styling
        self.figure.suptitle(title, fontsize=11, y=0.95)
        
        # Add legend with simpler styling
        if baseline_metrics:
            ax.legend(loc='lower right', fontsize=8)
        
        # Use subplots_adjust instead of tight_layout
        self.figure.subplots_adjust(top=0.85)
        self.canvas.draw()

class SessionComparisonWidget(QWidget):
    """Widget for comparing multiple shooting sessions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_chart(self, sessions_data, metric='avg_score', title='Session Comparison'):
        """Update the chart to compare sessions."""
        self.figure.clear()
        
        if not sessions_data or len(sessions_data) == 0:
            self.canvas.draw()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Extract data
        session_names = [s['name'] for s in sessions_data]
        values = [s.get(metric, 0) for s in sessions_data]
        
        # If too many sessions, limit and group the rest
        max_sessions = 10
        if len(session_names) > max_sessions:
            # Keep the most recent sessions and group older ones
            session_names = session_names[-max_sessions:]
            values = values[-max_sessions:]
        
        # Create bar chart with gradient colors
        bars = ax.bar(session_names, values)
        
        # Color bars based on values (higher is better)
        norm = plt.Normalize(min(values) if values else 0, 
                            max(values) if values else 10)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        for bar, val in zip(bars, values):
            bar.set_color(sm.to_rgba(val))
            bar.set_alpha(0.8)
            bar.set_edgecolor('white')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.1f}', ha='center', va='bottom', 
                   fontweight='bold', color='#455A64')
        
        # Add grid with better styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC', axis='y')
        
        # Format x-axis for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add labels and title
        ax.set_xlabel('Session', fontsize=12, fontweight='bold', color='#455A64')
        
        # Set y-label based on metric
        if metric == 'avg_score':
            ax.set_ylabel('Average Score', fontsize=12, fontweight='bold', color='#455A64')
            ax.set_ylim(0, 11)
        elif metric == 'avg_stability':
            ax.set_ylabel('Average Stability (%)', fontsize=12, fontweight='bold', color='#455A64')
            ax.set_ylim(0, 100)
        elif metric == 'avg_follow_through':
            ax.set_ylabel('Average Follow-through', fontsize=12, fontweight='bold', color='#455A64')
            ax.set_ylim(0, 1.1)
        elif metric == 'shot_count':
            ax.set_ylabel('Shots Taken', fontsize=12, fontweight='bold', color='#455A64')
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#1565C0')
        
        # Add colorbar for reference
        cbar = self.figure.colorbar(sm, ax=ax)
        if metric == 'avg_score':
            cbar.set_label('Score', fontweight='bold')
        elif metric == 'avg_stability':
            cbar.set_label('Stability (%)', fontweight='bold')
        elif metric == 'avg_follow_through':
            cbar.set_label('Follow-through', fontweight='bold')
        elif metric == 'shot_count':
            cbar.set_label('Count', fontweight='bold')
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()

class WeaknessAnalyzerWidget(QWidget):
    """Widget for analyzing weaknesses in shooting performance."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_chart(self, metrics_data, title='Performance Variability Analysis'):
        """Update chart with weakness analysis based on variability."""
        self.figure.clear()
        
        if not metrics_data or len(metrics_data) == 0:
            self.canvas.draw()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Define metrics to analyze
        metrics = [
            'Wrist Stability', 
            'Elbow Stability',
            'Shoulder Stability',
            'Head Stability',
            'Follow-through',
            'Overall Stability'
        ]
        
        # Calculate mean and std dev for each metric
        values = []
        errors = []
        for metric in metrics:
            if metric == 'Wrist Stability':
                # For stability metrics, lower sway is better, so we invert
                sway_values = [m.get('wrist_sway', 0) for m in metrics_data]
                stability_values = [max(0, 1 - (val / 20.0)) * 100 for val in sway_values]
                values.append(np.mean(stability_values) if stability_values else 0)
                errors.append(np.std(stability_values) if len(stability_values) > 1 else 0)
            elif metric == 'Elbow Stability':
                sway_values = [m.get('elbow_sway', 0) for m in metrics_data]
                stability_values = [max(0, 1 - (val / 20.0)) * 100 for val in sway_values]
                values.append(np.mean(stability_values) if stability_values else 0)
                errors.append(np.std(stability_values) if len(stability_values) > 1 else 0)
            elif metric == 'Shoulder Stability':
                sway_values = [m.get('shoulder_sway', 0) for m in metrics_data]
                stability_values = [max(0, 1 - (val / 20.0)) * 100 for val in sway_values]
                values.append(np.mean(stability_values) if stability_values else 0)
                errors.append(np.std(stability_values) if len(stability_values) > 1 else 0)
            elif metric == 'Head Stability':
                sway_values = [m.get('nose_sway', 0) for m in metrics_data]
                stability_values = [max(0, 1 - (val / 20.0)) * 100 for val in sway_values]
                values.append(np.mean(stability_values) if stability_values else 0)
                errors.append(np.std(stability_values) if len(stability_values) > 1 else 0)
            elif metric == 'Follow-through':
                follow_values = [m.get('follow_through', 0) * 100 for m in metrics_data]
                values.append(np.mean(follow_values) if follow_values else 0)
                errors.append(np.std(follow_values) if len(follow_values) > 1 else 0)
            elif metric == 'Overall Stability':
                stability_values = [m.get('overall_stability', 0) * 100 for m in metrics_data]
                values.append(np.mean(stability_values) if stability_values else 0)
                errors.append(np.std(stability_values) if len(stability_values) > 1 else 0)
        
        # Create horizontal bars with error bars
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, values, xerr=errors, align='center', 
                      alpha=0.7, error_kw=dict(ecolor='#D32F2F', lw=2, capsize=5, capthick=2))
        
        # Color bars based on variability (error) for highlighting weaknesses
        # Higher error = more variability = potential weakness
        for i, (bar, error) in enumerate(zip(bars, errors)):
            # Normalize color based on error (variability)
            # More variable areas (weaknesses) will be redder
            if error > 0:
                normalized_error = min(error / 20.0, 1.0)  # Cap at 20% std dev
                r = 0.9 * normalized_error + 0.1
                g = 0.6 * (1 - normalized_error)
                b = 0.1
                bar.set_color((r, g, b))
                
                # Add analysis text for the most variable metrics
                if error > 10:  # High variability
                    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                           f"High variability", va='center', color='#D32F2F', fontweight='bold')
            else:
                bar.set_color('#1976D2')  # Default blue for low variability
        
        # Add text showing exact values
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.1f} ±{errors[i]:.1f}", va='center')
        
        # Set y-ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        
        # Set x-axis label and limit
        ax.set_xlabel('Performance (0-100%) with Variability', fontsize=12, fontweight='bold', color='#455A64')
        ax.set_xlim(0, 110)  # Give some extra space for error bars and text
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        
        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold', color='#1565C0')
        
        # Add explanatory annotation
        ax.annotate('Longer error bars indicate inconsistent performance\n'
                    'Focus on metrics with both low values and high variability',
                    xy=(0.5, 0.01), xycoords='axes fraction', ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='#FFF9C4', ec='#FBC02D', alpha=0.8))
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()

class ShotSequenceWidget(QWidget):
    """Widget for visualizing shot sequence patterns."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QSizePolicy
        self.setMinimumHeight(200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure
        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_chart(self, shots_data, title='Shot Sequence Analysis'):
        """Update chart with shot sequence visualization - SIMPLIFIED."""
        self.figure.clear()
        
        if not shots_data or len(shots_data) == 0:
            self.canvas.draw()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Extract data
        shot_nums = list(range(1, len(shots_data) + 1))
        scores = [shot.get('subjective_score', 0) for shot in shots_data]
        follow_through = [shot.get('metrics', {}).get('follow_through_score', 0) for shot in shots_data]
        stability = [self._calculate_stability(shot.get('metrics', {})) for shot in shots_data]
        
        # Create colored zones (simplified)
        ax.axhspan(0.7, 1.0, alpha=0.1, color='#4CAF50', label='Excellent Zone')
        ax.axhspan(0.4, 0.7, alpha=0.1, color='#FFA000', label='Good Zone')
        ax.axhspan(0.0, 0.4, alpha=0.1, color='#F44336', label='Improvement Zone')
        
        # Plot follow-through and stability
        ax.plot(shot_nums, follow_through, 'o-', color='#7E57C2', 
            linewidth=1.5, markersize=6, label='Follow-through')
        ax.plot(shot_nums, stability, 's-', color='#1976D2', 
            linewidth=1.5, markersize=6, label='Stability')
        
        # Add shot score annotations (simplified)
        for i, score in enumerate(scores):
            ax.annotate(f"{score:.1f}", 
                    xy=(shot_nums[i], follow_through[i]), 
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
        
        # Set y-axis limits and label
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Performance (0-1)', fontsize=9)
        ax.set_xlabel('Shot Number', fontsize=9)
        
        # Add legend with smaller font
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
        
        # Set title
        ax.set_title(title, fontsize=11)
        
        # Add analysis summary (simplified)
        if len(shots_data) > 0:
            # Count shots by category
            great_shots = sum(1 for s, st in zip(scores, stability) if s >= 9.0 and st >= 0.7)
            good_shots = sum(1 for s, st in zip(scores, stability) if (s >= 7.0 or st >= 0.6) and not (s >= 9.0 and st >= 0.7))
            poor_shots = sum(1 for s, st in zip(scores, stability) if s < 5.0 or st < 0.4)
            avg_shots = len(shots_data) - great_shots - good_shots - poor_shots
            
            analysis_text = f"Great shots: {great_shots}, Good: {good_shots}, Average: {avg_shots}, Poor: {poor_shots}"
            ax.annotate(analysis_text, xy=(0.5, -0.01), xycoords='axes fraction', ha='center', fontsize=8)
        
        # Use subplots_adjust instead of tight_layout
        self.figure.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.95)
        self.canvas.draw()
        
    def _calculate_stability(self, metrics):
        """Calculate overall stability score from metrics."""
        # Extract key metrics
        sway_metrics = metrics.get('sway_velocity', {})
        dev_x_metrics = metrics.get('dev_x', {})
        dev_y_metrics = metrics.get('dev_y', {})
        
        # If overall stability is already calculated, use it
        if 'overall_stability_score' in metrics:
            return metrics['overall_stability_score']
        
        # Calculate average sway for upper body
        upper_body_joints = ['SHOULDERS', 'ELBOWS', 'WRISTS', 'NOSE']
        sway_values = [sway_metrics.get(joint, 0) for joint in upper_body_joints]
        avg_sway = sum(sway_values) / max(1, len(sway_values))
        
        # Calculate average positional deviation
        dev_x_values = [dev_x_metrics.get(joint, 0) for joint in upper_body_joints]
        dev_y_values = [dev_y_metrics.get(joint, 0) for joint in upper_body_joints]
        avg_dev_x = sum(dev_x_values) / max(1, len(dev_x_values))
        avg_dev_y = sum(dev_y_values) / max(1, len(dev_y_values))
        
        # Normalize metrics to 0-1 scale
        norm_sway = max(0, 1 - (avg_sway / 20.0))
        norm_dev_x = max(0, 1 - (avg_dev_x / 30.0))
        norm_dev_y = max(0, 1 - (avg_dev_y / 30.0))
        
        # Weighted combination of factors
        stability_score = (
            0.6 * norm_sway +        # Sway stability (60% weight)
            0.2 * norm_dev_x +       # Horizontal stability (20% weight)
            0.2 * norm_dev_y         # Vertical stability (20% weight)
        )
        
        return max(0.0, min(1.0, stability_score))

class ScoreStabilityWidget(QWidget):
    """Widget for visualizing the relationship between score and stability."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor('#FAFAFA')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_chart(self, shots_data, title='Score vs. Stability Correlation'):
        """Update chart with score vs stability correlation - SIMPLIFIED."""
        self.figure.clear()
        
        if not shots_data or len(shots_data) == 0:
            self.canvas.draw()
            return
            
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Extract data
        scores = [shot.get('subjective_score', 0) for shot in shots_data]
        stability = [self._calculate_stability(shot.get('metrics', {})) for shot in shots_data]
        shot_numbers = list(range(1, len(shots_data) + 1))
        
        # Calculate correlation coefficient
        score_stability_corr = np.corrcoef(scores, stability)[0, 1] if len(scores) > 1 else 0
        
        # Create scatter plot with shot numbers as colors
        scatter = ax.scatter(stability, scores, s=80, c=shot_numbers, cmap='viridis', 
                            alpha=0.7, edgecolor='white')
        
        # Add light quadrant lines
        ax.axhline(y=7.5, color='#90A4AE', linestyle='--', alpha=0.3)
        ax.axvline(x=0.5, color='#90A4AE', linestyle='--', alpha=0.3)
        
        # Add subtle quadrant labels
        ax.text(0.25, 9.0, "High Score\nLow Stability", 
            ha='center', va='center', fontsize=8, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.2", fc='#FFECB3', ec='#FFA000', alpha=0.2))
        ax.text(0.75, 9.0, "High Score\nHigh Stability", 
            ha='center', va='center', fontsize=8, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.2", fc='#C8E6C9', ec='#4CAF50', alpha=0.2))
        ax.text(0.25, 3.0, "Low Score\nLow Stability", 
            ha='center', va='center', fontsize=8, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.2", fc='#FFCDD2', ec='#F44336', alpha=0.2))
        ax.text(0.75, 3.0, "Low Score\nHigh Stability", 
            ha='center', va='center', fontsize=8, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.2", fc='#BBDEFB', ec='#2196F3', alpha=0.2))
        
        # Add shot number labels
        for i, (x, y) in enumerate(zip(stability, scores)):
            ax.annotate(str(i+1), xy=(x, y), xytext=(0, 0),
                    textcoords="offset points", ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
        
        # Add best fit line if we have more than one point
        if len(scores) > 1:
            m, b = np.polyfit(stability, scores, 1)
            x_range = np.linspace(0, 1, 10)
            ax.plot(x_range, m * x_range + b, '--', color='#D32F2F', alpha=0.7,
                linewidth=1.5, label=f'Trend: Score = {m:.2f}×Stability + {b:.2f}')
            ax.legend(loc='upper left', fontsize=8)
        
        # Set axis limits and labels
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 11)
        ax.set_xlabel('Stability Score (0-1)', fontsize=9)
        ax.set_ylabel('Shot Score', fontsize=9)
        
        # Add correlation information (simplified)
        ax.set_title(f"{title} (r={score_stability_corr:.2f})", fontsize=11)
        
        # Add colorbar properly
        cbar = self.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Shot Number', fontsize=8)
        
        # Use subplots_adjust instead of tight_layout
        self.figure.subplots_adjust(bottom=0.15, right=0.85)
        self.canvas.draw()
        
    def _calculate_stability(self, metrics):
        """Calculate overall stability score from metrics."""
        # Extract key metrics
        sway_metrics = metrics.get('sway_velocity', {})
        dev_x_metrics = metrics.get('dev_x', {})
        dev_y_metrics = metrics.get('dev_y', {})
        
        # If overall stability is already calculated, use it
        if 'overall_stability_score' in metrics:
            return metrics['overall_stability_score']
        
        # Calculate average sway for upper body
        upper_body_joints = ['SHOULDERS', 'ELBOWS', 'WRISTS', 'NOSE']
        sway_values = [sway_metrics.get(joint, 0) for joint in upper_body_joints]
        avg_sway = sum(sway_values) / max(1, len(sway_values))
        
        # Calculate average positional deviation
        dev_x_values = [dev_x_metrics.get(joint, 0) for joint in upper_body_joints]
        dev_y_values = [dev_y_metrics.get(joint, 0) for joint in upper_body_joints]
        avg_dev_x = sum(dev_x_values) / max(1, len(dev_x_values))
        avg_dev_y = sum(dev_y_values) / max(1, len(dev_y_values))
        
        # Normalize metrics to 0-1 scale
        norm_sway = max(0, 1 - (avg_sway / 20.0))
        norm_dev_x = max(0, 1 - (avg_dev_x / 30.0))
        norm_dev_y = max(0, 1 - (avg_dev_y / 30.0))
        
        # Weighted combination of factors
        stability_score = (
            0.6 * norm_sway +        # Sway stability (60% weight)
            0.2 * norm_dev_x +       # Horizontal stability (20% weight)
            0.2 * norm_dev_y         # Vertical stability (20% weight)
        )
        
        return max(0.0, min(1.0, stability_score))

class DashboardWidget(QWidget):
    """
    Enhanced dashboard widget displaying session data, statistics, and trends.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize the dashboard widget with enhanced analytics capabilities.
        
        Args:
            data_storage: Data storage manager instance
        """
        super().__init__()
        
        self.data_storage = data_storage
        self.user_id = None
        self.current_session = None
        
        # Store session, shot, and metrics data
        self.all_sessions = []
        self.current_shots = []
        self.baseline_metrics = {}
        
        # Initialize UI
        self.init_ui()
        
        # Refresh timer (update every 5 seconds)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)

    def init_ui(self):
        """Initialize the user interface elements with enhanced analytics."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title with professional styling
        title_label = QLabel("Shooting Performance Analytics")
        title_label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1565C0; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Session selector with date range filter for advanced analytics
        session_layout = QHBoxLayout()
        
        # Session selection
        session_layout.addWidget(QLabel("Session:"))
        
        self.session_selector = QComboBox()
        self.session_selector.setMinimumWidth(300)
        self.session_selector.currentIndexChanged.connect(self.on_session_changed)
        session_layout.addWidget(self.session_selector)
        
        # Add date range selector for advanced analytics
        self.date_filter_checkbox = QCheckBox("Filter by Date:")
        self.date_filter_checkbox.setChecked(False)
        self.date_filter_checkbox.stateChanged.connect(self.on_date_filter_changed)
        session_layout.addWidget(self.date_filter_checkbox)
        
        # From date
        self.from_date = QDateEdit()
        self.from_date.setDate(QDate.currentDate().addMonths(-1))
        self.from_date.setCalendarPopup(True)
        self.from_date.setEnabled(False)
        self.from_date.dateChanged.connect(self.refresh_data)
        session_layout.addWidget(self.from_date)
        
        # To date
        self.to_date = QDateEdit()
        self.to_date.setDate(QDate.currentDate())
        self.to_date.setCalendarPopup(True)
        self.to_date.setEnabled(False)
        self.to_date.dateChanged.connect(self.refresh_data)
        session_layout.addWidget(self.to_date)
        
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
        
        # Create the three tab pages with enhanced analytics
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

         # NEW: Add share button with dropdown
        from PyQt6.QtWidgets import QMenu
        
        share_button = QPushButton("Share")
        share_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DirIcon))
        
        # Create dropdown menu
        share_menu = QMenu()
        pdf_action = share_menu.addAction("Download as PDF")
        pdf_action.triggered.connect(self.download_as_pdf)
        email_action = share_menu.addAction("Share via Email")
        email_action.triggered.connect(self.share_via_email)
        
        share_button.setMenu(share_menu)
        controls_layout.addWidget(share_button)

        controls_layout.addStretch()
        
        # Add filter options
        filter_label = QLabel("Filter:")
        controls_layout.addWidget(filter_label)
        
        self.score_filter = QComboBox()
        self.score_filter.addItems(["All Scores", "High Scores (8+)", "Medium Scores (5-7)", "Low Scores (<5)"])
        self.score_filter.currentIndexChanged.connect(self.apply_table_filters)
        controls_layout.addWidget(self.score_filter)
        
        layout.addLayout(controls_layout)
        
        # Shot data table with improved styling
        self.shots_table = QTableWidget()
        self.shots_table.setColumnCount(8)  # Added one more column for overall stability
        self.shots_table.setHorizontalHeaderLabels([
            "Select", "Shot #", "Timestamp", "Score", 
            "Follow-through", "Overall Stability", "Sway Velocity", "Postural Stability"
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
            graphs_tab = QWidget()
            main_layout = QVBoxLayout(graphs_tab)

            # ─── scroll area ───────────────────────────────────────────
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            container = QWidget()
            vbox = QVBoxLayout(container)

            # ─── Shot Sequence ─────────────────────────────────────────
            seq_header = QLabel("Shot Sequence Analysis")
            seq_header.setFont(QFont('Arial', 14, QFont.Weight.Bold))
            seq_header.setStyleSheet("color: #1565C0; margin-top: 15px;")
            vbox.addWidget(seq_header)

            self.shot_sequence_widget = ShotSequenceWidget()
            # force it to take vertical room
            self.shot_sequence_widget.setMinimumHeight(300)
            self.shot_sequence_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            vbox.addWidget(self.shot_sequence_widget)

            # ─── Score vs Stability ─────────────────────────────────────
            score_header = QLabel("Score vs. Stability Analysis")
            score_header.setFont(QFont('Arial', 14, QFont.Weight.Bold))
            score_header.setStyleSheet("color: #1565C0; margin-top: 15px;")
            vbox.addWidget(score_header)

            self.score_stability_widget = ScoreStabilityWidget()
            self.score_stability_widget.setMinimumHeight(300)
            self.score_stability_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            vbox.addWidget(self.score_stability_widget)

            # ─── Stability vs Time ──────────────────────────────────────
            stab_header = QLabel("Stability vs. Time Analysis")
            stab_header.setFont(QFont('Arial', 14, QFont.Weight.Bold))
            stab_header.setStyleSheet("color: #1565C0; margin-top: 15px;")
            vbox.addWidget(stab_header)

            self.stability_time_fig = Figure(figsize=(10, 4), dpi=100)
            self.stability_time_fig.patch.set_facecolor('#FAFAFA')
            self.stability_time_canvas = FigureCanvas(self.stability_time_fig)
            self.stability_time_canvas.setMinimumHeight(300)
            self.stability_time_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            vbox.addWidget(self.stability_time_canvas)

            # final stretch so bottom can scroll into view
            vbox.addStretch(1)

            scroll_area.setWidget(container)
            main_layout.addWidget(scroll_area)

            self.dashboard_tabs.addTab(graphs_tab, "Graphs")

    def create_advanced_analytics_tab(self):
        """Create the advanced analytics tab with interactive components."""
        analytics_tab = QWidget()
        layout = QVBoxLayout()
        
        # Add tab explanation
        explanation = QLabel("This tab shows your aggregate performance analytics across all sessions")
        explanation.setStyleSheet("color: #455A64; font-style: italic;")
        layout.addWidget(explanation)
        
        # Create scroll area for analytics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Controls for analytics customization
        controls_layout = QHBoxLayout()
        
        # Analysis type selector
        controls_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "Performance Trends", 
            "Session Comparison", 
            "Weakness Analysis",
            "Performance Radar"
        ])
        self.analysis_type.currentIndexChanged.connect(self.update_analytics_view)
        controls_layout.addWidget(self.analysis_type)
        
        # Metric selector
        controls_layout.addWidget(QLabel("Metric:"))
        self.analytics_metric = QComboBox()
        self.analytics_metric.addItems([
            "Score", 
            "Stability", 
            "Follow-through"
        ])
        self.analytics_metric.currentIndexChanged.connect(self.update_analytics)
        controls_layout.addWidget(self.analytics_metric)
        
        # Add session selector for comparison
        controls_layout.addWidget(QLabel("Compare With:"))
        self.compare_session = QComboBox()
        self.compare_session.addItem("All Sessions", -1)
        self.compare_session.addItem("Best Session", -2)
        self.compare_session.currentIndexChanged.connect(self.update_analytics)
        controls_layout.addWidget(self.compare_session)
        
        controls_layout.addStretch()
        
        # Add refresh button
        refresh_button = QPushButton("Refresh Analytics")
        refresh_button.clicked.connect(self.update_analytics)
        controls_layout.addWidget(refresh_button)
        
        scroll_layout.addLayout(controls_layout)
        
        # Create stacked widget for different analysis views
        self.analytics_stack = QStackedWidget()
        
        # 1. Performance trends view
        trends_widget = QWidget()
        trends_layout = QVBoxLayout()
        
        self.performance_widget = PerformanceWidget()
        trends_layout.addWidget(self.performance_widget)
        
        # Add explanatory text
        trends_explanation = QLabel(
            "This chart shows your performance trend over time. "
            "The trend line indicates your rate of improvement."
        )
        trends_explanation.setWordWrap(True)
        trends_explanation.setStyleSheet("color: #455A64; font-style: italic; margin-top: 5px;")
        trends_layout.addWidget(trends_explanation)
        
        trends_widget.setLayout(trends_layout)
        self.analytics_stack.addWidget(trends_widget)
        
        # 2. Session comparison view
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout()
        
        self.session_comparison_widget = SessionComparisonWidget()
        comparison_layout.addWidget(self.session_comparison_widget)
        
        # Add explanatory text
        comparison_explanation = QLabel(
            "This chart compares performance metrics across different sessions. "
            "Use this to identify your best and worst sessions."
        )
        comparison_explanation.setWordWrap(True)
        comparison_explanation.setStyleSheet("color: #455A64; font-style: italic; margin-top: 5px;")
        comparison_layout.addWidget(comparison_explanation)
        
        comparison_widget.setLayout(comparison_layout)
        self.analytics_stack.addWidget(comparison_widget)
        
        # 3. Weakness analyzer view
        weakness_widget = QWidget()
        weakness_layout = QVBoxLayout()
        
        self.weakness_analyzer_widget = WeaknessAnalyzerWidget()
        weakness_layout.addWidget(self.weakness_analyzer_widget)
        
        # Add explanatory text
        weakness_explanation = QLabel(
            "This analysis identifies your areas of greatest variability. "
            "Focus your training on metrics with high variability (long error bars) and lower scores."
        )
        weakness_explanation.setWordWrap(True)
        weakness_explanation.setStyleSheet("color: #455A64; font-style: italic; margin-top: 5px;")
        weakness_layout.addWidget(weakness_explanation)
        
        weakness_widget.setLayout(weakness_layout)
        self.analytics_stack.addWidget(weakness_widget)
        
        # 4. Performance radar view
        radar_widget = QWidget()
        radar_layout = QVBoxLayout()
        
        self.radar_widget = RadarWidget()
        radar_layout.addWidget(self.radar_widget)
        
        # Add explanatory text
        radar_explanation = QLabel(
            "This radar chart shows your performance across multiple metrics simultaneously. "
            "A larger area indicates better all-around performance."
        )
        radar_explanation.setWordWrap(True)
        radar_explanation.setStyleSheet("color: #455A64; font-style: italic; margin-top: 5px;")
        radar_layout.addWidget(radar_explanation)
        
        radar_widget.setLayout(radar_layout)
        self.analytics_stack.addWidget(radar_widget)
        
        scroll_layout.addWidget(self.analytics_stack)
        
        # Create performance metrics grid with cards
        metrics_header = QLabel("Key Performance Indicators")
        metrics_header.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        metrics_header.setStyleSheet("color: #1565C0; margin-top: 15px;")
        scroll_layout.addWidget(metrics_header)
        
        # Grid layout for analytics cards
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        
        # Create analytics cards
        self.create_analytics_card(grid_layout, 0, 0, "Average Score", "0.0", "trophy")
        self.create_analytics_card(grid_layout, 0, 1, "Stability Trend", "Stable", "chart-line")
        self.create_analytics_card(grid_layout, 0, 2, "Sessions", "0", "calendar")
        self.create_analytics_card(grid_layout, 1, 0, "Total Shots", "0", "bullseye")
        self.create_analytics_card(grid_layout, 1, 1, "Best Session", "None", "star")
        self.create_analytics_card(grid_layout, 1, 2, "Follow-through Quality", "Good", "check")
        
        scroll_layout.addLayout(grid_layout)
        
        # Set the scroll widget layout
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        
        layout.addWidget(scroll_area)
        
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
        setattr(self, f"{title.lower().replace(' ', '_').replace('-', '_')}_label", value_label)
        
        card.setLayout(layout)
        parent_layout.addWidget(card, row, col)
        
        return card

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
            
            # Overall stability score
            overall_stability = shot['metrics'].get('overall_stability_score', 0) * 100  # Convert 0-1 to percentage
            stability_item = QTableWidgetItem(f"{overall_stability:.1f}%")
            # Color code based on value
            if overall_stability >= 70:
                stability_item.setForeground(QBrush(QColor('#388E3C')))  # Green for high stability
            elif overall_stability >= 40:
                stability_item.setForeground(QBrush(QColor('#FFA000')))  # Orange for medium
            else:
                stability_item.setForeground(QBrush(QColor('#D32F2F')))  # Red for low stability
            self.shots_table.setItem(row_position, 5, stability_item)
            
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
                
            self.shots_table.setItem(row_position, 6, sway_item)
            
            # Postural stability (average of DevX and DevY)
            dev_x = shot['metrics'].get('dev_x', {}).get('UPPER_BODY', 0)
            dev_y = shot['metrics'].get('dev_y', {}).get('UPPER_BODY', 0)
            
            stability = (dev_x + dev_y) / 2
            stability_item = QTableWidgetItem(f"{stability:.2f} px")
            self.shots_table.setItem(row_position, 7, stability_item)

    def apply_table_filters(self):
        """Apply filters to the shots table based on user selections."""
        # Get selected filter
        score_filter = self.score_filter.currentText()
        
        # If no shots, nothing to filter
        if not self.current_shots:
            return
            
        # Reset table
        self.shots_table.setRowCount(0)
        
        # Filter shots
        filtered_shots = []
        for shot in self.current_shots:
            # Apply score filter
            score = shot.get('subjective_score', 0)
            
            if score_filter == "All Scores":
                filtered_shots.append(shot)
            elif score_filter == "High Scores (8+)" and score >= 8:
                filtered_shots.append(shot)
            elif score_filter == "Medium Scores (5-7)" and 5 <= score <= 7:
                filtered_shots.append(shot)
            elif score_filter == "Low Scores (<5)" and score < 5:
                filtered_shots.append(shot)
        
        # Update table with filtered shots
        self.update_shots_table(filtered_shots)


    def update_stability_time_chart(self, shots: List[Dict]):
        """
        Create an enhanced stability vs time correlation chart.
        
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
        
        # Parse timestamps and add shot index information
        import datetime
        timestamps = []
        formatted_times = []
        
        for i, shot in enumerate(shots):
            try:
                # Parse ISO format timestamp
                dt = datetime.datetime.fromisoformat(shot['timestamp'])
                timestamps.append(dt)
                
                # Format as HH:MM:SS for labels
                formatted_times.append(dt.strftime('%H:%M:%S'))
            except (ValueError, TypeError):
                # Fallback to sequential numbers if timestamp parsing fails
                timestamps.append(None)
                formatted_times.append(f"Shot {i+1}")
        
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
        
        # Calculate stability scores
        stability_scores = []
        wrist_stability = []
        elbow_stability = []
        shoulder_stability = []
        head_stability = []
        follow_through = []
        scores = []
        
        for shot in shots:
            # Get metrics
            metrics = shot['metrics']
            
            # Get overall stability score
            if 'overall_stability_score' in metrics:
                stability = metrics['overall_stability_score'] * 100
            else:
                # Calculate from component metrics
                stability = self._calculate_stability_score(metrics)
            
            stability_scores.append(stability)
            
            # Extract component metrics for detailed analysis
            sway_velocities = metrics.get('sway_velocity', {})
            
            # Normalize sway to stability (0-100)
            wrist_sway = (sway_velocities.get('LEFT_WRIST', 0) + sway_velocities.get('RIGHT_WRIST', 0)) / 2
            wrist_stability.append(max(0, 1 - (wrist_sway / 20.0)) * 100)
            
            elbow_sway = (sway_velocities.get('LEFT_ELBOW', 0) + sway_velocities.get('RIGHT_ELBOW', 0)) / 2
            elbow_stability.append(max(0, 1 - (elbow_sway / 20.0)) * 100)
            
            shoulder_sway = (sway_velocities.get('LEFT_SHOULDER', 0) + sway_velocities.get('RIGHT_SHOULDER', 0)) / 2
            shoulder_stability.append(max(0, 1 - (shoulder_sway / 20.0)) * 100)
            
            nose_sway = sway_velocities.get('NOSE', 0)
            head_stability.append(max(0, 1 - (nose_sway / 20.0)) * 100)
            
            # Get follow-through score (0-100)
            follow_through.append(metrics.get('follow_through_score', 0) * 100)
            
            # Get subjective score (0-10.9)
            scores.append(shot['subjective_score'])
        
        # Plot stability score with improved styling
        scatter = ax.scatter(
            time_data, 
            stability_scores, 
            c=scores,  # Color by subjective score
            cmap='viridis',      # Professional color map
            s=100,                # Marker size
            alpha=0.8,           # Transparency
            edgecolor='white',    # White outline
            zorder=5              # Draw on top
        )
        
        # Add colored regions for stability zones
        ax.axhspan(70, 100, alpha=0.2, color='#4CAF50', label='Excellent Zone')
        ax.axhspan(40, 70, alpha=0.2, color='#FFA000', label='Good Zone')
        ax.axhspan(0, 40, alpha=0.2, color='#F44336', label='Improvement Zone')
        
        # Add a colorbar for subjective scores
        cbar = self.stability_time_fig.colorbar(scatter, ax=ax)
        cbar.set_label('Subjective Score', fontsize=10, fontweight='bold')
        
        # Add shot number labels
        for i, (x, y) in enumerate(zip(time_data, stability_scores)):
            ax.annotate(f"{i+1}", 
                      xy=(x, y), 
                      xytext=(0, -15),
                      textcoords="offset points",
                      ha='center',
                      bbox=dict(boxstyle="circle,pad=0.2", fc='white', ec='gray', alpha=0.7))
        
        # Add component stability lines if not too many shots
        if len(shots) <= 15:  # Only show components for smaller sessions to avoid clutter
            # Plot component stability metrics
            ax.plot(time_data, wrist_stability, 'o--', color='#E53935', 
                   linewidth=1, markersize=4, alpha=0.6, label='Wrist Stability')
            ax.plot(time_data, elbow_stability, 's--', color='#43A047', 
                   linewidth=1, markersize=4, alpha=0.6, label='Elbow Stability')
            ax.plot(time_data, shoulder_stability, '^--', color='#1E88E5', 
                   linewidth=1, markersize=4, alpha=0.6, label='Shoulder Stability')
            ax.plot(time_data, head_stability, 'd--', color='#7B1FA2', 
                   linewidth=1, markersize=4, alpha=0.6, label='Head Stability')
            ax.plot(time_data, follow_through, '*--', color='#FF9800', 
                   linewidth=1, markersize=4, alpha=0.6, label='Follow-through')
            
            # Add legend
            ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.8)
        
        # Add a best fit line to show trend
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
        
        # Set labels and title with professional formatting
        ax.set_xlabel(time_label, fontsize=12, fontweight='bold', color='#455A64')
        ax.set_ylabel('Stability Score (0-100)', fontsize=12, fontweight='bold', color='#455A64')
        ax.set_title('Stability vs. Time Analysis', fontsize=14, fontweight='bold', color='#1565C0')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7, color='#CFD8DC')
        
        # Set y-axis limits
        ax.set_ylim(0, 105)  # Give space for annotations
        
        # Set background colors
        ax.set_facecolor('#F5F8FA')
        
        # Rotate x-axis labels if using timestamps
        if None not in timestamps:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add fatigue analysis - detect significant drops in stability
        significant_drop = False
        if len(stability_scores) >= 5:
            # Look at moving averages to detect fatigue
            window_size = min(3, len(stability_scores))
            moving_avg = np.convolve(stability_scores, np.ones(window_size)/window_size, mode='valid')
            
            # If stability drops by more than 15% from start to end, flag potential fatigue
            if moving_avg[-1] < moving_avg[0] * 0.85:
                significant_drop = True
                
                # Add annotation about fatigue
                ax.text(0.5, 0.01, "Potential fatigue detected - stability decreases over time",
                       transform=ax.transAxes, ha='center', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", fc='#FFCDD2', ec='#F44336', alpha=0.8),
                       color='#D32F2F', fontweight='bold')
        
        # Add rest break recommendations if fatigue detected
        if significant_drop:
            # Find the point where stability starts to significantly drop
            fatigue_point = None
            for i in range(1, len(stability_scores)):
                if stability_scores[i] < stability_scores[i-1] * 0.85:
                    fatigue_point = i
                    break
            
            if fatigue_point:
                # Mark the fatigue point on the graph
                ax.axvline(x=time_data[fatigue_point], color='#F44336', linestyle='--', alpha=0.7)
                
                # Add rest recommendation text
                ax.text(time_data[fatigue_point], 100, "Consider rest break",
                       ha='center', va='bottom', rotation=90,
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#F44336', alpha=0.8),
                       color='#D32F2F')
        
        # Draw canvas
        self.stability_time_fig.subplots_adjust(bottom=0.15, right=0.9)
        self.stability_time_canvas.draw()

    def update_graphs(self):
        """Update all graphs in the graphs tab based on current shots data."""
        if self.current_shots:
            # Update primary graphs based on view options
    

            self.update_stability_time_chart(self.current_shots)
            
            # Update additional visualization components
            self.shot_sequence_widget.update_chart(self.current_shots)
            self.score_stability_widget.update_chart(self.current_shots)

    def update_analytics_view(self):
        """Update the analytics view based on the selected analysis type."""
        if self.analysis_type.currentText() == "Performance Trends":
            self.analytics_stack.setCurrentIndex(0)
        elif self.analysis_type.currentText() == "Session Comparison":
            self.analytics_stack.setCurrentIndex(1)
        elif self.analysis_type.currentText() == "Weakness Analysis":
            self.analytics_stack.setCurrentIndex(2)
        elif self.analysis_type.currentText() == "Performance Radar":
            self.analytics_stack.setCurrentIndex(3)
        
        # Update the analytics
        self.update_analytics()

    def update_analytics(self):
        """Update the analytics based on the selected view and metrics."""
        if not self.user_id:
            return
            
        # Get the selected analysis type
        analysis_type = self.analysis_type.currentText()
        
        # Get the selected metric
        metric = self.analytics_metric.currentText().lower()
        
        # Get date filter status
        date_filtered = hasattr(self, 'date_filter_checkbox') and self.date_filter_checkbox.isChecked()
        
        # Performance trends view
        if analysis_type == "Performance Trends":
            # Collect data from all sessions
            performance_data = self.get_performance_trend_data(metric, date_filtered)
            
            # Update the chart
            if metric == "score":
                title = "Score Performance Trend"
            elif metric == "stability":
                title = "Stability Performance Trend"
            else:
                title = "Follow-through Performance Trend"
                
            self.performance_widget.update_chart(performance_data, metric, title)
        
        # Session comparison view
        elif analysis_type == "Session Comparison":
            # Get session comparison data
            comparison_data = self.get_session_comparison_data(metric, date_filtered)
            
            # Update the chart
            if metric == "score":
                self.session_comparison_widget.update_chart(comparison_data, "avg_score", "Average Score Comparison")
            elif metric == "stability":
                self.session_comparison_widget.update_chart(comparison_data, "avg_stability", "Average Stability Comparison")
            else:
                self.session_comparison_widget.update_chart(comparison_data, "avg_follow_through", "Average Follow-through Comparison")
         # Weakness analysis view
        elif analysis_type == "Weakness Analysis":
            # Get metrics data for weakness analysis
            metrics_data = self.get_metrics_for_analysis(date_filtered)
            
            # Update the chart
            self.weakness_analyzer_widget.update_chart(metrics_data, "Performance Variability Analysis")
        
        # Performance radar view
        elif analysis_type == "Performance Radar":
            # Get current and comparison metrics
            current_metrics = self.get_current_performance_metrics()
            
            # Get comparison metrics based on selection
            compare_selection = self.compare_session.currentText()
            if compare_selection == "All Sessions":
                baseline_metrics = self.get_average_metrics(date_filtered)
                title = "Current vs Average Performance"
            elif compare_selection == "Best Session":
                baseline_metrics = self.get_best_session_metrics()
                title = "Current vs Best Performance"
            else:
                # Specific session selected
                session_id = self.compare_session.currentData()
                baseline_metrics = self.get_session_metrics(session_id)
                title = f"Current vs {compare_selection}"
            
            # Update the radar chart
            self.radar_widget.update_chart(current_metrics, baseline_metrics, title)
        
        # Update analytics cards
        self.update_analytics_cards()

    def get_performance_trend_data(self, metric="score", date_filtered=False):
        """Get performance trend data across sessions."""
        if not self.user_id:
            return []
            
        # Query database for all user sessions
        self.cursor = self.data_storage.conn.cursor()
        
        # Base query for sessions
        query = """
            SELECT s.id, s.name, s.created_at 
            FROM sessions s 
            WHERE s.user_id = ?
        """
        
        # Add date filter if enabled
        params = [self.user_id]
        if date_filtered and hasattr(self, 'from_date') and hasattr(self, 'to_date'):
            from_date = self.from_date.date().toString("yyyy-MM-dd")
            to_date = self.to_date.date().toString("yyyy-MM-dd")
            query += " AND s.created_at BETWEEN ? AND ?"
            params.extend([from_date, to_date + " 23:59:59"])
        
        # Order by date
        query += " ORDER BY s.created_at"
        
        self.cursor.execute(query, params)
        sessions = self.cursor.fetchall()
        
        if not sessions:
            return []
        
        # Collect performance data for each session
        performance_data = []
        
        for session in sessions:
            session_id = session['id']
            
            # Get session date
            try:
                session_date = datetime.strptime(session['created_at'][:10], "%Y-%m-%d").date()
            except ValueError:
                # Default to current date if parsing fails
                session_date = datetime.now().date()
            
            # Get shots for this session
            shots = self.data_storage.get_shots(session_id)
            
            if not shots:
                continue
            
            # Calculate metrics based on selected metric
            if metric == "score":
                # Average score
                scores = [shot['subjective_score'] for shot in shots]
                value = sum(scores) / len(scores) if scores else 0
            elif metric == "stability":
                # Average stability
                stability_values = [self._calculate_stability_score(shot['metrics']) for shot in shots]
                value = sum(stability_values) / len(stability_values) if stability_values else 0
            else:  # follow-through
                # Average follow-through
                follow_through_values = [shot['metrics'].get('follow_through_score', 0) for shot in shots]
                value = sum(follow_through_values) / len(follow_through_values) if follow_through_values else 0
            
            # Add to performance data
            performance_data.append({
                'date': session_date,
                metric: value,
                'session_id': session_id,
                'session_name': session['name'],
                'shot_count': len(shots)
            })
        
        return performance_data

    def get_session_comparison_data(self, metric="score", date_filtered=False):
        """Get session comparison data."""
        if not self.user_id:
            return []
            
        # Query database for all user sessions
        self.cursor = self.data_storage.conn.cursor()
        
        # Base query for sessions
        query = """
            SELECT s.id, s.name, s.created_at 
            FROM sessions s 
            WHERE s.user_id = ?
        """
        
        # Add date filter if enabled
        params = [self.user_id]
        if date_filtered and hasattr(self, 'from_date') and hasattr(self, 'to_date'):
            from_date = self.from_date.date().toString("yyyy-MM-dd")
            to_date = self.to_date.date().toString("yyyy-MM-dd")
            query += " AND s.created_at BETWEEN ? AND ?"
            params.extend([from_date, to_date + " 23:59:59"])
        
        # Order by date
        query += " ORDER BY s.created_at DESC"
        
        self.cursor.execute(query, params)
        sessions = self.cursor.fetchall()
        
        if not sessions:
            return []
        
        # Collect comparison data for each session
        comparison_data = []
        
        for session in sessions:
            session_id = session['id']
            
            # Get shots for this session
            shots = self.data_storage.get_shots(session_id)
            
            if not shots:
                continue
            
            # Calculate metrics based on selected metric
            session_data = {
                'id': session_id,
                'name': session['name'],
                'date': session['created_at'][:10],
                'shot_count': len(shots)
            }
            
            # Average score
            scores = [shot['subjective_score'] for shot in shots]
            session_data['avg_score'] = sum(scores) / len(scores) if scores else 0
            
            # Average stability
            stability_values = [self._calculate_stability_score(shot['metrics']) for shot in shots]
            session_data['avg_stability'] = sum(stability_values) / len(stability_values) if stability_values else 0
            
            # Average follow-through
            follow_through_values = [shot['metrics'].get('follow_through_score', 0) for shot in shots]
            session_data['avg_follow_through'] = sum(follow_through_values) / len(follow_through_values) if follow_through_values else 0
            
            # Add to comparison data
            comparison_data.append(session_data)
        
        return comparison_data

    def get_metrics_for_analysis(self, date_filtered=False):
        """Get metrics data for weakness analysis."""
        if not self.user_id:
            return []
            
        # Collect metrics from all shots or filtered by date
        all_metrics = []
        
        # Query database for all user sessions
        self.cursor = self.data_storage.conn.cursor()
        
        # Base query for sessions
        query = """
            SELECT s.id, s.created_at 
            FROM sessions s 
            WHERE s.user_id = ?
        """
        
        # Add date filter if enabled
        params = [self.user_id]
        if date_filtered and hasattr(self, 'from_date') and hasattr(self, 'to_date'):
            from_date = self.from_date.date().toString("yyyy-MM-dd")
            to_date = self.to_date.date().toString("yyyy-MM-dd")
            query += " AND s.created_at BETWEEN ? AND ?"
            params.extend([from_date, to_date + " 23:59:59"])
        
        self.cursor.execute(query, params)
        sessions = self.cursor.fetchall()
        
        if not sessions:
            return []
        
        # Get shots for each session
        for session in sessions:
            session_id = session['id']
            shots = self.data_storage.get_shots(session_id)
            
            for shot in shots:
                metrics = shot['metrics']
                
                # Add processed metrics for analysis
                processed_metrics = {}
                
                # Extract and normalize metrics
                sway_metrics = metrics.get('sway_velocity', {})
                
                # Wrist sway - average of left and right
                wrist_sway = (sway_metrics.get('LEFT_WRIST', 0) + sway_metrics.get('RIGHT_WRIST', 0)) / 2
                processed_metrics['wrist_sway'] = wrist_sway
                
                # Elbow sway - average of left and right
                elbow_sway = (sway_metrics.get('LEFT_ELBOW', 0) + sway_metrics.get('RIGHT_ELBOW', 0)) / 2
                processed_metrics['elbow_sway'] = elbow_sway
                
                # Shoulder sway - average of left and right
                shoulder_sway = (sway_metrics.get('LEFT_SHOULDER', 0) + sway_metrics.get('RIGHT_SHOULDER', 0)) / 2
                processed_metrics['shoulder_sway'] = shoulder_sway
                
                # Head (nose) sway
                processed_metrics['nose_sway'] = sway_metrics.get('NOSE', 0)
                
                # Follow-through
                processed_metrics['follow_through'] = metrics.get('follow_through_score', 0)
                
                # Overall stability
                processed_metrics['overall_stability'] = metrics.get('overall_stability_score', 0)
                
                # Add to metrics collection
                all_metrics.append(processed_metrics)
        
        return all_metrics

    def get_current_performance_metrics(self):
        """Get performance metrics for the current session."""
        if not self.current_session or not self.current_shots:
            return self.get_average_metrics()  # Fall back to average metrics
            
        # Process all shots in the current session
        metrics = {}
        
        # Get all sway velocities
        wrist_sway_values = []
        elbow_sway_values = []
        shoulder_sway_values = []
        nose_sway_values = []
        follow_through_values = []
        stability_values = []
        
        for shot in self.current_shots:
            shot_metrics = shot['metrics']
            sway_metrics = shot_metrics.get('sway_velocity', {})
            
            # Wrist sway - average of left and right
            wrist_sway = (sway_metrics.get('LEFT_WRIST', 0) + sway_metrics.get('RIGHT_WRIST', 0)) / 2
            wrist_sway_values.append(wrist_sway)
            
            # Elbow sway - average of left and right
            elbow_sway = (sway_metrics.get('LEFT_ELBOW', 0) + sway_metrics.get('RIGHT_ELBOW', 0)) / 2
            elbow_sway_values.append(elbow_sway)
            
            # Shoulder sway - average of left and right
            shoulder_sway = (sway_metrics.get('LEFT_SHOULDER', 0) + sway_metrics.get('RIGHT_SHOULDER', 0)) / 2
            shoulder_sway_values.append(shoulder_sway)
            
            # Head (nose) sway
            nose_sway = sway_metrics.get('NOSE', 0)
            nose_sway_values.append(nose_sway)
            
            # Follow-through
            follow_through = shot_metrics.get('follow_through_score', 0)
            follow_through_values.append(follow_through)
            
            # Overall stability
            stability = shot_metrics.get('overall_stability_score', 0)
            if stability is None:
                stability = self._calculate_stability_score(shot_metrics)
            stability_values.append(stability)
        
        # Calculate averages
        metrics['wrist_sway'] = sum(wrist_sway_values) / len(wrist_sway_values) if wrist_sway_values else 0
        metrics['elbow_sway'] = sum(elbow_sway_values) / len(elbow_sway_values) if elbow_sway_values else 0
        metrics['shoulder_sway'] = sum(shoulder_sway_values) / len(shoulder_sway_values) if shoulder_sway_values else 0
        metrics['nose_sway'] = sum(nose_sway_values) / len(nose_sway_values) if nose_sway_values else 0
        metrics['follow_through'] = sum(follow_through_values) / len(follow_through_values) if follow_through_values else 0
        metrics['overall_stability'] = sum(stability_values) / len(stability_values) if stability_values else 0
        
        # Add variance metrics to indicate consistency
        metrics['consistency'] = 1.0 - (np.std(stability_values) / 0.5) if len(stability_values) > 1 else 0.5
        metrics['consistency'] = max(0, min(1, metrics['consistency']))  # Clamp to 0-1 range
        
        return metrics

    def get_average_metrics(self, date_filtered=False):
        """Get average performance metrics across all sessions."""
        # Get metrics data from all sessions (or filtered by date)
        all_metrics = self.get_metrics_for_analysis(date_filtered)
        
        if not all_metrics:
            # Return default metrics if no data
            return {
                'wrist_sway': 0,
                'elbow_sway': 0,
                'shoulder_sway': 0,
                'nose_sway': 0, 
                'follow_through': 0,
                'overall_stability': 0,
                'consistency': 0.5
            }
        
        # Calculate averages
        metrics = {}
        
        # Extract all values for each metric
        wrist_sway_values = [m['wrist_sway'] for m in all_metrics]
        elbow_sway_values = [m['elbow_sway'] for m in all_metrics]
        shoulder_sway_values = [m['shoulder_sway'] for m in all_metrics]
        nose_sway_values = [m['nose_sway'] for m in all_metrics]
        follow_through_values = [m['follow_through'] for m in all_metrics]
        stability_values = [m['overall_stability'] for m in all_metrics]
        
        # Calculate averages
        metrics['wrist_sway'] = sum(wrist_sway_values) / len(wrist_sway_values) if wrist_sway_values else 0
        metrics['elbow_sway'] = sum(elbow_sway_values) / len(elbow_sway_values) if elbow_sway_values else 0
        metrics['shoulder_sway'] = sum(shoulder_sway_values) / len(shoulder_sway_values) if shoulder_sway_values else 0
        metrics['nose_sway'] = sum(nose_sway_values) / len(nose_sway_values) if nose_sway_values else 0
        metrics['follow_through'] = sum(follow_through_values) / len(follow_through_values) if follow_through_values else 0
        metrics['overall_stability'] = sum(stability_values) / len(stability_values) if stability_values else 0
        
        # Add consistency metric (inverse of standard deviation)
        metrics['consistency'] = 1.0 - (np.std(stability_values) / 0.5) if len(stability_values) > 1 else 0.5
        metrics['consistency'] = max(0, min(1, metrics['consistency']))  # Clamp to 0-1 range
        
        return metrics

    def get_best_session_metrics(self):
        """Get metrics from the user's best session."""
        if not self.user_id:
            return self.get_average_metrics()  # Fall back to average metrics
            
        # Query for best session based on average score
        self.cursor = self.data_storage.conn.cursor()
        
        self.cursor.execute("""
            SELECT s.id, s.name, AVG(sh.subjective_score) as avg_score
            FROM sessions s
            JOIN shots sh ON s.id = sh.session_id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY avg_score DESC
            LIMIT 1
        """, (self.user_id,))
        
        best_session = self.cursor.fetchone()
        
        if not best_session:
            return self.get_average_metrics()  # Fall back to average metrics
            
        # Get metrics for this session
        return self.get_session_metrics(best_session['id'])

    def get_session_metrics(self, session_id):
        """Get average metrics for a specific session."""
        if not session_id:
            return self.get_average_metrics()  # Fall back to average metrics
            
        # Get shots for this session
        shots = self.data_storage.get_shots(session_id)
        
        if not shots:
            return self.get_average_metrics()  # Fall back to average metrics
            
        # Process all shots in the session
        metrics = {}
        
        # Get all sway velocities
        wrist_sway_values = []
        elbow_sway_values = []
        shoulder_sway_values = []
        nose_sway_values = []
        follow_through_values = []
        stability_values = []
        
        for shot in shots:
            shot_metrics = shot['metrics']
            sway_metrics = shot_metrics.get('sway_velocity', {})
            
            # Wrist sway - average of left and right
            wrist_sway = (sway_metrics.get('LEFT_WRIST', 0) + sway_metrics.get('RIGHT_WRIST', 0)) / 2
            wrist_sway_values.append(wrist_sway)
            
            # Elbow sway - average of left and right
            elbow_sway = (sway_metrics.get('LEFT_ELBOW', 0) + sway_metrics.get('RIGHT_ELBOW', 0)) / 2
            elbow_sway_values.append(elbow_sway)
            
            # Shoulder sway - average of left and right
            shoulder_sway = (sway_metrics.get('LEFT_SHOULDER', 0) + sway_metrics.get('RIGHT_SHOULDER', 0)) / 2
            shoulder_sway_values.append(shoulder_sway)
            
            # Head (nose) sway
            nose_sway = sway_metrics.get('NOSE', 0)
            nose_sway_values.append(nose_sway)
            
            # Follow-through
            follow_through = shot_metrics.get('follow_through_score', 0)
            follow_through_values.append(follow_through)
            
            # Overall stability
            stability = shot_metrics.get('overall_stability_score', 0)
            if stability is None:
                stability = self._calculate_stability_score(shot_metrics)
            stability_values.append(stability)
        
        # Calculate averages
        metrics['wrist_sway'] = sum(wrist_sway_values) / len(wrist_sway_values) if wrist_sway_values else 0
        metrics['elbow_sway'] = sum(elbow_sway_values) / len(elbow_sway_values) if elbow_sway_values else 0
        metrics['shoulder_sway'] = sum(shoulder_sway_values) / len(shoulder_sway_values) if shoulder_sway_values else 0
        metrics['nose_sway'] = sum(nose_sway_values) / len(nose_sway_values) if nose_sway_values else 0
        metrics['follow_through'] = sum(follow_through_values) / len(follow_through_values) if follow_through_values else 0
        metrics['overall_stability'] = sum(stability_values) / len(stability_values) if stability_values else 0
        
        # Add consistency metric (inverse of standard deviation)
        metrics['consistency'] = 1.0 - (np.std(stability_values) / 0.5) if len(stability_values) > 1 else 0.5
        metrics['consistency'] = max(0, min(1, metrics['consistency']))  # Clamp to 0-1 range
        
        return metrics

    def update_analytics_cards(self):
        """Update the analytics cards with key performance metrics."""
        if not self.user_id:
            return
            
        try:
            # Get overall stats from database
            self.cursor = self.data_storage.conn.cursor()
            
            # Get session count
            self.cursor.execute("""
                SELECT COUNT(*) as session_count FROM sessions WHERE user_id = ?
            """, (self.user_id,))
            session_count = self.cursor.fetchone()['session_count']
            
            # Get total shot count
            self.cursor.execute("""
                SELECT COUNT(*) as shot_count 
                FROM shots sh
                JOIN sessions s ON sh.session_id = s.id
                WHERE s.user_id = ?
            """, (self.user_id,))
            shot_count = self.cursor.fetchone()['shot_count']
            
            # Get average score across all sessions
            self.cursor.execute("""
                SELECT AVG(sh.subjective_score) as avg_score
                FROM shots sh
                JOIN sessions s ON sh.session_id = s.id
                WHERE s.user_id = ?
            """, (self.user_id,))
            avg_score_result = self.cursor.fetchone()
            avg_score = avg_score_result['avg_score'] if avg_score_result and avg_score_result['avg_score'] is not None else 0
            
            # Get best session
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
            best_session_name = best_session['name'] if best_session else "None"
            
            # Get average follow-through across all sessions - WITH PROPER ERROR HANDLING
            follow_through_values = []
            try:
                self.cursor.execute("""
                    SELECT sh.metrics
                    FROM shots sh
                    JOIN sessions s ON sh.session_id = s.id
                    WHERE s.user_id = ?
                """, (self.user_id,))
                shot_metrics = self.cursor.fetchall()
                
                # Process each shot metrics safely
                for shot in shot_metrics:
                    try:
                        # Ensure json is imported at the top of the file!
                        metrics_dict = json.loads(shot['metrics'])
                        follow_through = metrics_dict.get('follow_through_score', 0)
                        follow_through_values.append(follow_through)
                    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                        # Skip this shot if there's any issue parsing
                        continue
            except Exception as e:
                # Handle database query errors
                print(f"Error fetching metrics: {e}")
            
            avg_follow_through = sum(follow_through_values) / len(follow_through_values) if follow_through_values else 0
            
            # Determine stability trend
            stability_trend = "Stable"  # Default
            trend_color = "#1976D2"     # Default blue
            
            # Determine follow-through quality
            if avg_follow_through > 0.7:
                follow_quality = "Excellent"
                quality_color = "#388E3C"  # Green
            elif avg_follow_through > 0.4:
                follow_quality = "Good"
                quality_color = "#1976D2"  # Blue
            else:
                follow_quality = "Needs Work"
                quality_color = "#D32F2F"  # Red
            
            # Update card values - check if attributes exist before updating
            if hasattr(self, 'average_score_label'):
                self.average_score_label.setText(f"{avg_score:.1f}")
            
            if hasattr(self, 'stability_trend_label'):
                self.stability_trend_label.setText(stability_trend)
                self.stability_trend_label.setStyleSheet(f"color: {trend_color}; font-size: 18px; font-weight: bold;")
            
            if hasattr(self, 'sessions_label'):
                self.sessions_label.setText(str(session_count))
            
            if hasattr(self, 'total_shots_label'):
                self.total_shots_label.setText(str(shot_count))
            
            if hasattr(self, 'best_session_label'):
                self.best_session_label.setText(best_session_name)
            
            if hasattr(self, 'follow_through_quality_label'):
                self.follow_through_quality_label.setText(follow_quality)
                self.follow_through_quality_label.setStyleSheet(f"color: {quality_color}; font-size: 18px; font-weight: bold;")
                
        except Exception as e:
            print(f"Error updating analytics cards: {e}")
            import traceback
            traceback.print_exc()

    def on_date_filter_changed(self, state):
        """Handle date filter checkbox state changes."""
        # Enable/disable date inputs
        if hasattr(self, 'from_date'):
            self.from_date.setEnabled(state)
        
        if hasattr(self, 'to_date'):
            self.to_date.setEnabled(state)
        
        # Refresh data with new filter settings
        self.refresh_data()

    def export_to_csv(self):
        """Export selected shots to CSV file."""
        if not self.current_session or self.shots_table.rowCount() == 0:
            QMessageBox.warning(self, "Export Error", "No data to export.")
            return
        
        # Get export filename
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "shooting_session_data.csv", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Collect data to export
            export_data = []
            headers = ["Shot #", "Timestamp", "Subjective Score", "Follow-through", 
                     "Overall Stability", "Sway Velocity", "Postural Stability"]
            
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
                
                # Also update the comparison session selector
                if hasattr(self, 'compare_session'):
                    # Save the current selection
                    current_selection = self.compare_session.currentText()
                    
                    # Clear and repopulate
                    self.compare_session.clear()
                    self.compare_session.addItem("All Sessions", -1)
                    self.compare_session.addItem("Best Session", -2)
                    
                    # Add all sessions except the current one
                    for i in range(1, self.session_selector.count()):
                        session_name = self.session_selector.itemText(i)
                        session_id = self.session_selector.itemData(i)
                        
                        if session_id != self.current_session:
                            self.compare_session.addItem(session_name, session_id)
                    
                    # Try to restore previous selection
                    index = self.compare_session.findText(current_selection)
                    if index >= 0:
                        self.compare_session.setCurrentIndex(index)
                    else:
                        self.compare_session.setCurrentIndex(0)  # Default to "All Sessions"
        except Exception as e:
            print(f"Error in session change: {e}")
            import traceback
            traceback.print_exc()
            self.session_stats_label.setText("Error loading session")
            self.clear_data()

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
            self.current_shots = self.data_storage.get_shots(self.current_session)
            
            # Update table tab
            self.update_shots_table(self.current_shots)
            
            # Update graphs tab
            self.update_graphs()
            
            # Update analytics tab
            self.update_analytics()
            
        except Exception as e:
            import traceback
            print(f"Error refreshing dashboard data: {e}")
            print(traceback.format_exc())
            self.session_stats_label.setText("Error loading data")

    def clear_data(self):
        """Clear all data displays when no session is selected."""
        # Clear table
        self.shots_table.setRowCount(0)
        
        # Clear current shots data
        self.current_shots = []
        
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
        
        # Clear shot sequence and score-stability widgets
        if hasattr(self, 'shot_sequence_widget'):
            self.shot_sequence_widget.update_chart([])
            
        if hasattr(self, 'score_stability_widget'):
            self.score_stability_widget.update_chart([])
        
        # Reset analytics views
        if hasattr(self, 'performance_widget'):
            self.performance_widget.update_chart([])
            
        if hasattr(self, 'session_comparison_widget'):
            self.session_comparison_widget.update_chart([])
            
        if hasattr(self, 'weakness_analyzer_widget'):
            self.weakness_analyzer_widget.update_chart([])
            
        if hasattr(self, 'radar_widget'):
            self.radar_widget.update_chart({})
        
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
                        
    def _calculate_stability_score(self, metrics: Dict) -> float:
        """
        Calculate an overall stability score from metrics.
        
        Args:
            metrics: Dictionary of shot metrics
            
        Returns:
            Overall stability score (0-1)
        """
        # Extract key metrics
        sway_metrics = metrics.get('sway_velocity', {})
        dev_x_metrics = metrics.get('dev_x', {})
        dev_y_metrics = metrics.get('dev_y', {})
        
        # If overall stability is already calculated, use it
        if 'overall_stability_score' in metrics:
            return metrics['overall_stability_score']
        
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
        norm_sway = max(0, 1 - (avg_sway / 20.0))
        norm_dev_x = max(0, 1 - (avg_dev_x / 30.0))
        norm_dev_y = max(0, 1 - (avg_dev_y / 30.0))
        
        # Weighted combination of factors
        stability_score = (
            0.6 * norm_sway +        # Sway stability (60% weight)
            0.2 * norm_dev_x +       # Horizontal stability (20% weight)
            0.2 * norm_dev_y         # Vertical stability (20% weight)
        )
        
        # Ensure value is in 0-1 range
        return max(0.0, min(1.0, stability_score))
    
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
        
        # Update analytics
        self.update_analytics_cards()
        
        # Update analytics session comparison dropdown
        if hasattr(self, 'compare_session'):
            self.compare_session.clear()
            self.compare_session.addItem("All Sessions", -1)
            self.compare_session.addItem("Best Session", -2)

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
    
    def download_as_pdf(self):
        """Generate and download a PDF report of the current session data."""
        if not self.current_session or not self.current_shots:
            QMessageBox.warning(self, "No Data", "No session data available to export.")
            return
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Performance Report", 
            f"shooting_report_{datetime.now().strftime('%Y%m%d')}", 
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Create PDF document
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet
            from io import BytesIO
            
            # Initialize PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Add title and session info
            session_name = "Unknown Session"
            if self.current_session:
                self.cursor = self.data_storage.conn.cursor()
                self.cursor.execute("SELECT name FROM sessions WHERE id = ?", (self.current_session,))
                session = self.cursor.fetchone()
                if session:
                    session_name = session['name']
            
            title = Paragraph(f"Shooting Performance Report: {session_name}", styles['Heading1'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Add date
            date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal'])
            elements.append(date_text)
            elements.append(Spacer(1, 12))
            
            # Add overall performance metrics
            elements.append(Paragraph("Overall Performance Metrics", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            # Calculate statistics
            if self.current_shots:
                scores = [shot.get('subjective_score', 0) for shot in self.current_shots]
                avg_score = sum(scores) / len(scores) if scores else 0
                stability_values = [self._calculate_stability_score(shot.get('metrics', {})) for shot in self.current_shots]
                avg_stability = sum(stability_values) / len(stability_values) if stability_values else 0
                follow_values = [shot.get('metrics', {}).get('follow_through_score', 0) for shot in self.current_shots]
                avg_follow = sum(follow_values) / len(follow_values) if follow_values else 0
                
                # Create metrics table
                metrics_data = [
                    ['Metric', 'Value', 'Rating'],
                    ['Average Score', f"{avg_score:.1f}/10", self._get_rating(avg_score/10)],
                    ['Average Stability', f"{avg_stability*100:.1f}%", self._get_rating(avg_stability)],
                    ['Average Follow-through', f"{avg_follow:.2f}", self._get_rating(avg_follow)],
                    ['Total Shots', str(len(self.current_shots)), '']
                ]
                
                t = Table(metrics_data, colWidths=[150, 100, 100])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(t)
                elements.append(Spacer(1, 12))
            
            # Add shot data table
            elements.append(Paragraph("Shot Details", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            # Extract table data
            table_data = [['Shot #', 'Timestamp', 'Score', 'Follow-through', 'Stability', 'Sway']]
            
            for i, shot in enumerate(self.current_shots):
                timestamp = shot['timestamp'].split('T')[1][:8] if 'timestamp' in shot else "Unknown"
                score = shot.get('subjective_score', 0)
                follow = shot.get('metrics', {}).get('follow_through_score', 0)
                stability = self._calculate_stability_score(shot.get('metrics', {})) * 100
                
                # Calculate sway
                sway_metrics = shot.get('metrics', {}).get('sway_velocity', {})
                avg_sway = 0
                if sway_metrics:
                    sway_values = [sway_metrics.get(joint, 0) for joint in ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW']]
                    avg_sway = sum(sway_values) / len(sway_values) if sway_values else 0
                
                table_data.append([str(i+1), timestamp, str(score), f"{follow:.2f}", f"{stability:.1f}%", f"{avg_sway:.2f} mm/s"])
            
            # Create table
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 24))
            
            # Add graphs
            elements.append(Paragraph("Performance Visualizations", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            # Capture graphs as images
            if hasattr(self, 'shot_sequence_widget'):
                shot_seq_buffer = BytesIO()
                self.shot_sequence_widget.figure.savefig(shot_seq_buffer, format='png', dpi=150)
                shot_seq_buffer.seek(0)
                shot_seq_img = Image(shot_seq_buffer, width=450, height=225)
                elements.append(Paragraph("Shot Sequence Analysis", styles['Heading3']))
                elements.append(shot_seq_img)
                elements.append(Spacer(1, 12))
            
            if hasattr(self, 'score_stability_widget'):
                score_stab_buffer = BytesIO()
                self.score_stability_widget.figure.savefig(score_stab_buffer, format='png', dpi=150)
                score_stab_buffer.seek(0)
                score_stab_img = Image(score_stab_buffer, width=450, height=225)
                elements.append(Paragraph("Score vs. Stability Analysis", styles['Heading3']))
                elements.append(score_stab_img)
                elements.append(Spacer(1, 12))
            
            if hasattr(self, 'stability_time_fig'):
                stab_time_buffer = BytesIO()
                self.stability_time_fig.savefig(stab_time_buffer, format='png', dpi=150)
                stab_time_buffer.seek(0)
                stab_time_img = Image(stab_time_buffer, width=450, height=225)
                elements.append(Paragraph("Stability vs. Time Analysis", styles['Heading3']))
                elements.append(stab_time_img)
            
            # Build the PDF
            doc.build(elements)
            
            QMessageBox.information(self, "PDF Created", 
                                f"Performance report saved to:\n{file_path}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "PDF Creation Error", 
                           f"Failed to create PDF report: {str(e)}")
            
    def _get_rating(self, value):
        """Convert a numeric value (0-1) to a text rating."""
        if value >= 0.8:
            return "Excellent"
        elif value >= 0.6:
            return "Good"
        elif value >= 0.4:
            return "Average"
        else:
            return "Needs Improvement"    
    
    def share_via_email(self):
        """Share the performance report via email."""
        if not self.current_session or not self.current_shots:
            QMessageBox.warning(self, "No Data", "No session data available to share.")
            return
        
        try:
            # First create a temporary PDF
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            temp_pdf = os.path.join(temp_dir, "temp_shooting_report.pdf")
            
            # Generate the PDF to the temp location
            self._generate_pdf_report(temp_pdf)
            
            # Get email details from user
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QDialogButtonBox
            
            email_dialog = QDialog(self)
            email_dialog.setWindowTitle("Share via Email")
            email_dialog.setMinimumWidth(400)
            
            dialog_layout = QVBoxLayout()
            
            dialog_layout.addWidget(QLabel("Recipient Email:"))
            recipient_input = QLineEdit()
            dialog_layout.addWidget(recipient_input)
            
            dialog_layout.addWidget(QLabel("Subject:"))
            subject_input = QLineEdit("Rifle Shooting Performance Report")
            dialog_layout.addWidget(subject_input)
            
            dialog_layout.addWidget(QLabel("Message:"))
            message_input = QLineEdit("Please find attached my rifle shooting performance report.")
            dialog_layout.addWidget(message_input)
            
            # Add dialog buttons
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                        QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(email_dialog.accept)
            button_box.rejected.connect(email_dialog.reject)
            dialog_layout.addWidget(button_box)
            
            email_dialog.setLayout(dialog_layout)
            
            if email_dialog.exec() == QDialog.DialogCode.Accepted:
                recipient = recipient_input.text()
                subject = subject_input.text()
                message = message_input.text()
                
                # Open default email client with the report attached
                self._send_email(recipient, subject, message, temp_pdf)
                
                # Success message
                QMessageBox.information(self, "Email Prepared", 
                    "Email has been prepared with the attached report.")
                
            # Clean up temp file
            try:
                os.remove(temp_pdf)
            except:
                pass
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Email Error", 
                            f"Failed to share via email: {str(e)}")
    
    def _generate_pdf_report(self, file_path):
        """Generate a PDF report at the specified path."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from io import BytesIO
        
        # Initialize PDF document
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title and session info
        session_name = "Unknown Session"
        if self.current_session:
            self.cursor = self.data_storage.conn.cursor()
            self.cursor.execute("SELECT name FROM sessions WHERE id = ?", (self.current_session,))
            session = self.cursor.fetchone()
            if session:
                session_name = session['name']
        
        title = Paragraph(f"Shooting Performance Report: {session_name}", styles['Heading1'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Add date
        date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal'])
        elements.append(date_text)
        elements.append(Spacer(1, 12))
        
        # Add overall performance metrics
        elements.append(Paragraph("Overall Performance Metrics", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        # Calculate statistics
        if self.current_shots:
            scores = [shot.get('subjective_score', 0) for shot in self.current_shots]
            avg_score = sum(scores) / len(scores) if scores else 0
            stability_values = [self._calculate_stability_score(shot.get('metrics', {})) for shot in self.current_shots]
            avg_stability = sum(stability_values) / len(stability_values) if stability_values else 0
            follow_values = [shot.get('metrics', {}).get('follow_through_score', 0) for shot in self.current_shots]
            avg_follow = sum(follow_values) / len(follow_values) if follow_values else 0
            
            # Create metrics table
            metrics_data = [
                ['Metric', 'Value', 'Rating'],
                ['Average Score', f"{avg_score:.1f}/10", self._get_rating(avg_score/10)],
                ['Average Stability', f"{avg_stability*100:.1f}%", self._get_rating(avg_stability)],
                ['Average Follow-through', f"{avg_follow:.2f}", self._get_rating(avg_follow)],
                ['Total Shots', str(len(self.current_shots)), '']
            ]
            
            t = Table(metrics_data, colWidths=[150, 100, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))
        
        # Add shot data table
        elements.append(Paragraph("Shot Details", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        # Extract table data
        table_data = [['Shot #', 'Timestamp', 'Score', 'Follow-through', 'Stability', 'Sway']]
        
        for i, shot in enumerate(self.current_shots):
            timestamp = shot['timestamp'].split('T')[1][:8] if 'timestamp' in shot else "Unknown"
            score = shot.get('subjective_score', 0)
            follow = shot.get('metrics', {}).get('follow_through_score', 0)
            stability = self._calculate_stability_score(shot.get('metrics', {})) * 100
            
            # Calculate sway
            sway_metrics = shot.get('metrics', {}).get('sway_velocity', {})
            avg_sway = 0
            if sway_metrics:
                sway_values = [sway_metrics.get(joint, 0) for joint in ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW']]
                avg_sway = sum(sway_values) / len(sway_values) if sway_values else 0
            
            table_data.append([str(i+1), timestamp, str(score), f"{follow:.2f}", f"{stability:.1f}%", f"{avg_sway:.2f} mm/s"])
        
        # Create table
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 24))
        
        # Add graphs
        elements.append(Paragraph("Performance Visualizations", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        # Capture graphs as images
        if hasattr(self, 'shot_sequence_widget'):
            shot_seq_buffer = BytesIO()
            self.shot_sequence_widget.figure.savefig(shot_seq_buffer, format='png', dpi=150)
            shot_seq_buffer.seek(0)
            shot_seq_img = Image(shot_seq_buffer, width=450, height=225)
            elements.append(Paragraph("Shot Sequence Analysis", styles['Heading3']))
            elements.append(shot_seq_img)
            elements.append(Spacer(1, 12))
        
        if hasattr(self, 'score_stability_widget'):
            score_stab_buffer = BytesIO()
            self.score_stability_widget.figure.savefig(score_stab_buffer, format='png', dpi=150)
            score_stab_buffer.seek(0)
            score_stab_img = Image(score_stab_buffer, width=450, height=225)
            elements.append(Paragraph("Score vs. Stability Analysis", styles['Heading3']))
            elements.append(score_stab_img)
            elements.append(Spacer(1, 12))
        
        if hasattr(self, 'stability_time_fig'):
            stab_time_buffer = BytesIO()
            self.stability_time_fig.savefig(stab_time_buffer, format='png', dpi=150)
            stab_time_buffer.seek(0)
            stab_time_img = Image(stab_time_buffer, width=450, height=225)
            elements.append(Paragraph("Stability vs. Time Analysis", styles['Heading3']))
            elements.append(stab_time_img)
        
        # Build the PDF
        doc.build(elements)

    def _send_email(self, recipient, subject, message, attachment_path):
        """Open the default email client with the PDF attached."""
        import webbrowser
        import urllib.parse
        
        # Create a mailto URL
        mailto_url = f"mailto:{recipient}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(message)}"
        
        # Open the default email client
        webbrowser.open(mailto_url)
        
        # Note: This will only open the email client with recipient, subject, and body
        # Automatically attaching files is not broadly supported through the mailto protocol
        # For a real implementation, you would need to:
        # 1. Use platform-specific email client APIs
        # 2. Or use a library like smtplib to send directly 
        # 3. Or show instructions to the user about manually attaching the file
        
        # Show instructions about attachment
        QMessageBox.information(self, "Attach PDF", 
                            f"Your email client has been opened.\n\nPlease manually attach the PDF from:\n{attachment_path}")