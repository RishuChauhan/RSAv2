from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.ui import theme

class PlaceholderWidget(QWidget):
    """
    Reusable placeholder widget shown when no session is active.
    Displays a large icon, title, subtitle, and an optional action button.
    """

    # Signal emitted when action button is clicked
    action_clicked = pyqtSignal()

    def __init__(self, icon: str, title: str, subtitle: str, button_text: str = None):
        """
        Initialize the placeholder widget.

        Args:
            icon: Unicode icon character to display (large)
            title: Main title text
            subtitle: Subtitle description text
            button_text: Text for the action button (optional)
        """
        super().__init__()
        self.init_ui(icon, title, subtitle, button_text)

    def init_ui(self, icon, title, subtitle, button_text):
        """Initialize the UI layout."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Large icon
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 64px; color: {theme.TEXT_MUTED};")
        layout.addWidget(icon_label)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {theme.TEXT_SECONDARY};
        """)
        layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel(subtitle)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet(f"font-size: 13px; color: {theme.TEXT_MUTED};")
        layout.addWidget(subtitle_label)

        # Action button (optional)
        if button_text:
            button_layout = QHBoxLayout()
            button_layout.addStretch()

            self.action_button = QPushButton(button_text)
            self.action_button.setStyleSheet(theme.button_primary())
            self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.action_button.clicked.connect(self.action_clicked.emit)

            button_layout.addWidget(self.action_button)
            button_layout.addStretch()
            layout.addLayout(button_layout)

        self.setLayout(layout)
