from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFrame, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon

from src.ui import theme

class NavButton(QPushButton):
    """Custom navigation button for sidebar."""

    def __init__(self, text, icon_text, parent=None):
        super().__init__(parent)
        self.setText(f"{icon_text}  {text}")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(theme.NAV_BUTTON_STYLE)
        self.setFixedHeight(50)

class SidebarWidget(QWidget):
    """
    Vertical sidebar navigation widget.
    Replaces the top toolbar.
    """

    # Signal emitted when a navigation item is clicked
    # Argument: index of the page to switch to
    navigation_changed = pyqtSignal(int)

    # Signal for logout
    logout_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(theme.SIDEBAR_STYLE)

        self.current_user = None
        self.current_session = None

        self.init_ui()

    def init_ui(self):
        """Initialize the sidebar layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Logo Area ────────────────────────────────────────────────────────
        logo_container = QWidget()
        logo_container.setFixedHeight(100)
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        app_logo = QLabel("RSA")
        app_logo.setStyleSheet(f"""
            font-size: 32px;
            font-weight: bold;
            color: {theme.ACCENT_CYAN};
            letter-spacing: 2px;
        """)
        logo_layout.addWidget(app_logo)

        app_subtitle = QLabel("v2 · Shooting Analysis")
        app_subtitle.setStyleSheet(f"""
            font-size: 11px;
            color: {theme.TEXT_SECONDARY};
        """)
        logo_layout.addWidget(app_subtitle)

        layout.addWidget(logo_container)

        # ── Navigation Items ────────────────────────────────────────────────
        self.nav_buttons = []

        # 0: Dashboard
        self.btn_dashboard = self._add_nav_button("Dashboard", "📊", 0, layout)

        # 1: Live Analysis
        self.btn_live = self._add_nav_button("Live Analysis", "⚡", 1, layout)

        # 2: 3D Visualization
        self.btn_viz = self._add_nav_button("3D View", "🎯", 2, layout)

        # 3: Replay
        self.btn_replay = self._add_nav_button("Replay", "▶", 3, layout)

        # 4: Settings
        self.btn_settings = self._add_nav_button("Settings", "⚙", 4, layout)

        layout.addStretch()

        # ── Session Info Panel ──────────────────────────────────────────────
        self.session_card = QFrame()
        self.session_card.setStyleSheet(theme.card_style())
        self.session_card.setVisible(False)

        session_layout = QVBoxLayout(self.session_card)
        session_layout.setContentsMargins(10, 10, 10, 10)

        self.session_name_label = QLabel("Session Name")
        self.session_name_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-weight: bold;")
        session_layout.addWidget(self.session_name_label)

        self.shot_count_label = QLabel("0 shots")
        self.shot_count_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: {theme.FONT_SM};")
        session_layout.addWidget(self.shot_count_label)

        self.timer_label = QLabel("⏱ 00:00")
        self.timer_label.setStyleSheet(f"color: {theme.ACCENT_CYAN}; font-weight: bold; margin-top: 5px;")
        session_layout.addWidget(self.timer_label)

        layout.addWidget(self.session_card)

        # Spacer
        layout.addSpacing(20)

        # ── User Info Area ──────────────────────────────────────────────────
        user_container = QWidget()
        user_layout = QHBoxLayout(user_container)
        user_layout.setContentsMargins(15, 15, 15, 15)

        # User avatar (circle with initials)
        self.user_avatar = QLabel("U")
        self.user_avatar.setFixedSize(32, 32)
        self.user_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_avatar.setStyleSheet(f"""
            background-color: {theme.ACCENT_BLUE};
            color: white;
            border-radius: 16px;
            font-weight: bold;
        """)
        user_layout.addWidget(self.user_avatar)

        # User name and logout
        name_layout = QVBoxLayout()
        name_layout.setSpacing(2)

        self.user_name_label = QLabel("User")
        self.user_name_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-weight: 600;")
        name_layout.addWidget(self.user_name_label)

        logout_btn = QPushButton("Logout")
        logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        logout_btn.setStyleSheet(f"""
            background: transparent;
            text-align: left;
            color: {theme.TEXT_MUTED};
            padding: 0;
            border: none;
            font-size: {theme.FONT_SM};
        """)
        logout_btn.clicked.connect(self.logout_requested.emit)
        name_layout.addWidget(logout_btn)

        user_layout.addLayout(name_layout)
        layout.addWidget(user_container)

        self.setLayout(layout)

        # Session timer
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._update_timer)
        self.session_start_time = None

    def _add_nav_button(self, text, icon, index, layout):
        """Helper to add navigation button."""
        btn = NavButton(text, icon)
        btn.clicked.connect(lambda: self._handle_nav_click(index))
        layout.addWidget(btn)
        self.nav_buttons.append(btn)
        return btn

    def _handle_nav_click(self, index):
        """Handle navigation button click."""
        # Uncheck all buttons
        for btn in self.nav_buttons:
            btn.setChecked(False)

        # Check the clicked button
        self.nav_buttons[index].setChecked(True)

        # Emit signal
        self.navigation_changed.emit(index)

    def set_active_index(self, index):
        """Programmatically set the active navigation item."""
        if 0 <= index < len(self.nav_buttons):
            self._handle_nav_click(index)

    def set_user(self, user):
        """Set the current user info."""
        self.current_user = user
        if user:
            name = user.get('name', 'User')
            self.user_name_label.setText(name)

            # Set initials
            initials = "".join([n[0] for n in name.split()[:2]]).upper()
            self.user_avatar.setText(initials)
        else:
            self.user_name_label.setText("Guest")
            self.user_avatar.setText("?")

    def start_session(self, session_name):
        """Start a session and show the info card."""
        self.session_name_label.setText(session_name)
        self.shot_count_label.setText("0 shots")
        self.timer_label.setText("⏱ 00:00")

        self.session_card.setVisible(True)
        self.session_card.setStyleSheet(f"""
            background-color: {theme.SURFACE};
            border: 1px solid {theme.SUCCESS};
            border-radius: 10px;
            padding: 12px;
        """)

        import time
        self.session_start_time = time.time()
        self.session_timer.start(1000)

    def update_shot_count(self, count):
        """Update the shot count display."""
        self.shot_count_label.setText(f"{count} shots")

    def end_session(self):
        """End the current session."""
        self.session_timer.stop()
        self.session_card.setVisible(False)
        self.session_start_time = None

    def _update_timer(self):
        """Update the elapsed time display."""
        if self.session_start_time:
            import time
            elapsed = int(time.time() - self.session_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.timer_label.setText(f"⏱ {minutes:02d}:{seconds:02d}")
