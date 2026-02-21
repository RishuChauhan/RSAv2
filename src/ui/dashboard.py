from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QStackedWidget, QScrollArea, QGridLayout,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import logging
import contextlib

from src.data_storage import DataStorage
from src.ui import theme
from src.ui.placeholder_widget import PlaceholderWidget

logger = logging.getLogger(__name__)

class SummaryCard(QFrame):
    """Card displaying a key performance metric."""
    def __init__(self, title, value, trend=None, accent_color=theme.ACCENT_BLUE, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.BORDER};
                border-radius: 10px;
                border-left: 4px solid {accent_color};
            }}
        """)
        self.setFixedSize(200, 100)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 11px; border: none;")
        layout.addWidget(lbl_title)
        
        self.lbl_value = QLabel(value)
        self.lbl_value.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: 28px; font-weight: bold; border: none;")
        layout.addWidget(self.lbl_value)
        
        if trend:
            lbl_trend = QLabel(trend)
            # Simple heuristic for color
            color = theme.SUCCESS if "+" in trend else theme.DANGER if "-" in trend else theme.TEXT_MUTED
            lbl_trend.setStyleSheet(f"color: {color}; font-size: 12px; border: none;")
            layout.addWidget(lbl_trend)

    def update_value(self, value, trend=None):
        self.lbl_value.setText(value)
        # Update trend logic if needed

class DashboardWidget(QWidget):
    """
    Enhanced dashboard widget displaying session data, statistics, and trends.
    """
    
    def __init__(self, data_storage: DataStorage):
        super().__init__()
        self.data_storage = data_storage
        self.user_id = None
        self.current_session = None
        self.current_shots = []
        
        self.init_ui()
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)

    def init_ui(self):
        # Stacked Layout
        self.stack = QStackedWidget()

        # 1. Placeholder
        self.placeholder = PlaceholderWidget(
            icon="📊",
            title="Dashboard",
            subtitle="Select a session to view analytics.",
            button_text="Create New Session"
        )
        self.placeholder.action_clicked.connect(self.request_new_session)
        self.stack.addWidget(self.placeholder)

        # 2. Content
        self.content_widget = QWidget()
        self.init_content_ui()
        self.stack.addWidget(self.content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    def request_new_session(self):
        main = self.window()
        if hasattr(main, '_create_new_session'):
            main._create_new_session()

    def init_content_ui(self):
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # ── Summary Cards ────────────────────────────────────────────────────
        cards_layout = QHBoxLayout()
        self.card_shots = SummaryCard("Total Shots", "0", "+2", theme.ACCENT_BLUE)
        self.card_stability = SummaryCard("Avg Stability", "0%", "+1.2%", theme.SUCCESS)
        self.card_sway = SummaryCard("Avg Sway", "0.0", "-0.5", theme.WARNING)
        self.card_score = SummaryCard("Best Score", "0.0", "", theme.ACCENT_CYAN)

        cards_layout.addWidget(self.card_shots)
        cards_layout.addWidget(self.card_stability)
        cards_layout.addWidget(self.card_sway)
        cards_layout.addWidget(self.card_score)
        cards_layout.addStretch()

        layout.addLayout(cards_layout)

        # ── Session Selector Bar ─────────────────────────────────────────────
        selector_frame = QFrame()
        selector_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.SURFACE};
                border-radius: 8px;
            }}
        """)
        selector_frame.setFixedHeight(60)
        sel_layout = QHBoxLayout(selector_frame)

        sel_layout.addWidget(QLabel("Current Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(250)
        self.session_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {theme.DARK_BG};
                border: 1px solid {theme.BORDER};
                color: {theme.TEXT_PRIMARY};
                padding: 5px;
                border-radius: 4px;
            }}
        """)
        self.session_combo.currentIndexChanged.connect(self.on_session_changed)
        sel_layout.addWidget(self.session_combo)
        sel_layout.addStretch()

        layout.addWidget(selector_frame)

        # ── Charts Area ──────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        charts_container = QWidget()
        charts_layout = QVBoxLayout(charts_container)
        charts_layout.setSpacing(20)

        # Stability Trend Chart
        self.trend_figure = Figure(figsize=(8, 4), dpi=100)
        self.trend_canvas = FigureCanvas(self.trend_figure)
        self._style_chart(self.trend_figure)
        charts_layout.addWidget(self.trend_canvas)

        # Shot Distribution (Placeholder for now)
        self.dist_figure = Figure(figsize=(8, 4), dpi=100)
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self._style_chart(self.dist_figure)
        charts_layout.addWidget(self.dist_canvas)

        scroll.setWidget(charts_container)
        layout.addWidget(scroll)

    def _style_chart(self, fig):
        """Apply dark theme to matplotlib figure."""
        fig.patch.set_facecolor(theme.DARK_BG)

    def _plot_dark_chart(self, fig, x, y, title, ylabel):
        """Helper to plot with dark theme."""
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)

        # Plot data
        ax.plot(x, y, color=theme.ACCENT_CYAN, linewidth=2, marker='o')

        # Styling
        ax.set_title(title, color=theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, color=theme.TEXT_SECONDARY)
        ax.tick_params(colors=theme.TEXT_SECONDARY)

        # Grid
        ax.grid(True, color=theme.BORDER, linestyle='--', alpha=0.5)

        # Spines
        for spine in ax.spines.values():
            spine.set_edgecolor(theme.BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        fig.canvas.draw()

    def set_user(self, user_id):
        self.user_id = user_id
        self.refresh_sessions()

    def refresh_sessions(self):
        if not self.user_id: return
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItem("Select Session...", -1)

        sessions = self.data_storage.get_sessions(self.user_id)
        for s in sessions:
            self.session_combo.addItem(f"{s['name']} ({s['created_at'][:10]})", s['id'])
        self.session_combo.blockSignals(False)

    def set_session(self, session_id):
        self.current_session = session_id
        if session_id:
            self.stack.setCurrentIndex(1)
            # Update combo selection
            idx = self.session_combo.findData(session_id)
            if idx >= 0: self.session_combo.setCurrentIndex(idx)
            self.refresh_data()
        else:
            self.stack.setCurrentIndex(0)

    def on_session_changed(self, index):
        sid = self.session_combo.itemData(index)
        if sid > 0:
            self.set_session(sid)
        else:
            self.set_session(None)

    def refresh_data(self):
        if not self.current_session: return

        # Fetch data
        shots = self.data_storage.get_shots(self.current_session)
        self.current_shots = shots
        stats = self.data_storage.get_session_stats(self.current_session)

        # Update Cards
        self.card_shots.update_value(str(len(shots)))

        # Calculate Avg Stability (mock calculation for demo)
        avg_stab = 0
        if shots:
            # Assume stability is stored or calculate it
            pass
        self.card_stability.update_value(f"{avg_stab}%")

        best = stats.get('max_subjective_score', 0)
        self.card_score.update_value(str(best))

        # Update Charts
        if shots:
            x = range(1, len(shots) + 1)
            y = [s['subjective_score'] for s in shots]
            self._plot_dark_chart(self.trend_figure, x, y, "Score Trend", "Score")

            # Second chart: Stability Trend (mock data)
            y_stab = [s.get('metrics', {}).get('overall_stability_score', 0) * 100 for s in shots]
            self._plot_dark_chart(self.dist_figure, x, y_stab, "Stability Trend", "Stability %")
