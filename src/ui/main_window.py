from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QStackedWidget, QMessageBox,
    QLineEdit, QToolBar, QStatusBar, QFrame, QTabWidget
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QAction, QPainter, QColor, QLinearGradient, QGradient

import os
import hashlib
import sys

# Import UI components
from src.ui import theme
from src.ui.sidebar import SidebarWidget
from src.ui.dashboard import DashboardWidget
from src.ui.live_analysis import LiveAnalysisWidget
from src.ui.visualization import VisualizationWidget
from src.ui.replay import ReplayWidget
from src.ui.settings import SettingsWidget

# Import other modules
from src.data_storage import DataStorage

class MainWindow(QMainWindow):
    """Main application window for the rifle shooting analysis application."""
    
    def __init__(self):
        """Initialize the main window and UI components."""
        super().__init__()
        
        # Initialize data storage
        self.data_storage = DataStorage()
        
        # User authentication state
        self.current_user = None
        self.current_session = None
        
        # Set up the main window
        self.setWindowTitle("RSA v2 — Rifle Shooting Analysis")
        self.setMinimumSize(1280, 800)
        
        # Apply global theme
        self.setStyleSheet(theme.APP_STYLESHEET)

        # Create stacked widget for authentication vs main app
        self.root_stack = QStackedWidget()
        self.setCentralWidget(self.root_stack)
        
        # Create login/register widget
        self.auth_widget = self._create_auth_widget()
        self.root_stack.addWidget(self.auth_widget)

        # Create main app container (Sidebar + Content)
        self.main_container = QWidget()
        self.main_layout = QHBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.navigation_changed.connect(self._handle_navigation)
        self.sidebar.logout_requested.connect(self._handle_logout)
        self.main_layout.addWidget(self.sidebar)

        # Content Stack
        self.content_stack = QStackedWidget()
        self.main_layout.addWidget(self.content_stack)
        
        # Add container to root stack
        self.root_stack.addWidget(self.main_container)

        # Initialize content widgets
        self._init_content_widgets()
        
        # Start with authentication screen
        self.root_stack.setCurrentIndex(0)
        
        # Create status bar
        self._init_statusbar()

    def _init_statusbar(self):
        """Initialize the custom status bar."""
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {theme.SURFACE};
                color: {theme.TEXT_SECONDARY};
                border-top: 1px solid {theme.BORDER};
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Right side permanent widgets
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 10, 0)

        # Analysis status dot
        self.analysis_status = QLabel("")
        self.analysis_status.setTextFormat(Qt.TextFormat.RichText)
        self.analysis_status.setVisible(False)
        right_layout.addWidget(self.analysis_status)

        # Separator
        self.status_separator = QLabel(" · ")
        self.status_separator.setVisible(False)
        right_layout.addWidget(self.status_separator)

        # User name
        self.status_user_label = QLabel("")
        self.status_user_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        right_layout.addWidget(self.status_user_label)

        self.status_bar.addPermanentWidget(right_widget)

    def _init_content_widgets(self):
        """Initialize and add content widgets to the stack."""
        # 0: Dashboard
        self.dashboard_widget = DashboardWidget(self.data_storage)
        self.content_stack.addWidget(self.dashboard_widget)
        
        # 1: Live Analysis
        self.live_analysis_widget = LiveAnalysisWidget(self.data_storage)
        self.content_stack.addWidget(self.live_analysis_widget)

        # 2: 3D Visualization
        self.visualization_widget = VisualizationWidget(self.data_storage)
        self.content_stack.addWidget(self.visualization_widget)

        # 3: Replay
        self.replay_widget = ReplayWidget(self.data_storage)
        self.content_stack.addWidget(self.replay_widget)

        # 4: Settings
        self.settings_widget = SettingsWidget(self.data_storage)
        self.content_stack.addWidget(self.settings_widget)

    def _create_auth_widget(self) -> QWidget:
        """Create professionally styled authentication widget."""
        auth_widget = QWidget()
        layout = QHBoxLayout(auth_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Left Panel: Branding ─────────────────────────────────────────────
        branding_panel = QFrame()
        branding_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.SURFACE};
                border-right: 1px solid {theme.BORDER};
            }}
        """)
        branding_layout = QVBoxLayout(branding_panel)
        branding_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        branding_layout.setSpacing(20)

        # App Title
        title = QLabel("RSA")
        title.setStyleSheet(f"font-size: 72px; font-weight: bold; color: {theme.ACCENT_CYAN};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        branding_layout.addWidget(title)

        # Tagline
        tagline = QLabel("Precision shooting analytics")
        tagline.setStyleSheet(f"font-size: 18px; color: {theme.TEXT_SECONDARY};")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        branding_layout.addWidget(tagline)

        # Features list
        features_widget = QWidget()
        features_layout = QVBoxLayout(features_widget)
        features_layout.setSpacing(15)

        features = [
            "Real-time stability metrics",
            "Advanced 3D pose estimation",
            "Session history & trends"
        ]

        for feature in features:
            f_label = QLabel(f"✓  {feature}")
            f_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: 14px;")
            features_layout.addWidget(f_label)

        branding_layout.addWidget(features_widget)
        
        layout.addWidget(branding_panel, stretch=1)

        # ── Right Panel: Login Form ──────────────────────────────────────────
        form_panel = QWidget()
        form_layout = QVBoxLayout(form_panel)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Form Container Card
        card = QFrame()
        card.setFixedWidth(400)
        card.setStyleSheet(theme.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(20)
        card_layout.setContentsMargins(30, 40, 30, 40)

        # Tab Widget for Login/Register
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{
                background: transparent;
                color: {theme.TEXT_SECONDARY};
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {theme.ACCENT_BLUE};
                border-bottom: 2px solid {theme.ACCENT_BLUE};
            }}
            QTabBar::tab:hover {{ color: {theme.TEXT_PRIMARY}; }}
        """)
        
        # Login Tab
        login_tab = QWidget()
        login_form = QVBoxLayout(login_tab)
        login_form.setSpacing(15)

        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Email")
        self.login_email.setStyleSheet(theme.INPUT_STYLE)
        login_form.addWidget(self.login_email)

        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.login_password.setStyleSheet(theme.INPUT_STYLE)
        login_form.addWidget(self.login_password)

        login_btn = QPushButton("Login")
        login_btn.setStyleSheet(theme.button_primary())
        login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        login_btn.clicked.connect(lambda: self._handle_login(self.login_email.text(), self.login_password.text()))
        login_form.addWidget(login_btn)

        tab_widget.addTab(login_tab, "Login")

        # Register Tab
        register_tab = QWidget()
        register_form = QVBoxLayout(register_tab)
        register_form.setSpacing(15)

        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText("Full Name")
        self.reg_name.setStyleSheet(theme.INPUT_STYLE)
        register_form.addWidget(self.reg_name)

        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("Email")
        self.reg_email.setStyleSheet(theme.INPUT_STYLE)
        register_form.addWidget(self.reg_email)

        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Password")
        self.reg_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_password.setStyleSheet(theme.INPUT_STYLE)
        register_form.addWidget(self.reg_password)

        reg_btn = QPushButton("Create Account")
        reg_btn.setStyleSheet(theme.button_primary())
        reg_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reg_btn.clicked.connect(
            lambda: self._handle_register(
                self.reg_name.text(),
                self.reg_email.text(),
                self.reg_password.text()
            )
        )
        register_form.addWidget(reg_btn)
        
        tab_widget.addTab(register_tab, "Register")
        
        card_layout.addWidget(tab_widget)
        form_layout.addWidget(card)
        
        layout.addWidget(form_panel, stretch=1)
        
        return auth_widget

    def _handle_navigation(self, index):
        """Switch content stack based on sidebar navigation."""
        self.content_stack.setCurrentIndex(index)

    def _handle_login(self, email: str, password: str):
        """Handle user login."""
        if not email or not password:
            QMessageBox.warning(self, "Login Error", "Please enter email and password.")
            return
        
        self.status_bar.showMessage("Authenticating...")
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Authenticate
        user = self.data_storage.authenticate_user(email, password_hash)
        
        if user:
            self.current_user = user
            self.status_bar.showMessage(f"Logged in as {user['name']}")

            # Switch to main app
            self.root_stack.setCurrentIndex(1)
            
            # Update sidebar user info
            self.sidebar.set_user(user)
            self.sidebar.set_active_index(0)  # Default to Dashboard
            
            # Update status bar user info
            self.status_user_label.setText(f"👤 {user['name']}")
            
            # Propagate user to all widgets
            self.dashboard_widget.set_user(user['id'])
            self.live_analysis_widget.set_user(user['id'])
            self.visualization_widget.set_user(user['id'])
            self.replay_widget.set_user(user['id'])
            self.settings_widget.set_user(user['id'])
            
            # Check for existing session
            self._check_for_active_session(user['id'])
            
        else:
            self.status_bar.showMessage("Login failed")
            QMessageBox.critical(self, "Login Error", "Invalid email or password.")

    def _handle_register(self, name: str, email: str, password: str):
        """Handle user registration."""
        if not name or not email or not password:
            QMessageBox.warning(self, "Registration Error", "Please fill all fields.")
            return
        
        if '@' not in email or '.' not in email:
            QMessageBox.warning(self, "Registration Error", "Please enter a valid email.")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "Registration Error", "Password must be at least 6 characters.")
            return
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        user_id = self.data_storage.create_user(name, email, password_hash)
        
        if user_id > 0:
            QMessageBox.information(self, "Success", "Account created. Please log in.")
            # Switch to login tab
            self.auth_widget.findChild(QTabWidget).setCurrentIndex(0)
        else:
            QMessageBox.critical(self, "Error", "Registration failed. Email may depend in use.")

    def _handle_logout(self):
        """Handle logout."""
        self.current_user = None
        self.current_session = None
        self.root_stack.setCurrentIndex(0)
        self.sidebar.set_user(None)
        self.status_bar.showMessage("Logged out")
        
        # Reset window title
        self.setWindowTitle("RSA v2 — Rifle Shooting Analysis")

    def _check_for_active_session(self, user_id):
        """Check and load the last active session."""
        self.cursor = self.data_storage.conn.cursor()
        self.cursor.execute(
            """SELECT s.id, s.name FROM sessions s
            WHERE s.user_id = ?
            ORDER BY s.created_at DESC LIMIT 1""",
            (user_id,)
        )
        last_session = self.cursor.fetchone()
        
        if last_session:
            self.current_session = {
                'id': last_session['id'],
                'name': last_session['name']
            }
            # Update sidebar
            self.sidebar.start_session(last_session['name'])

            # Update widgets
            self.dashboard_widget.set_session(last_session['id'])
            self.live_analysis_widget.set_session(last_session['id'])
            self.visualization_widget.set_session(last_session['id'])
            self.replay_widget.set_session(last_session['id'])

            # Update window title
            self.setWindowTitle(f"RSA v2 — {last_session['name']}")

    # Exposed methods for session management from widgets
    def start_new_session(self, name):
        """Called when a new session is created from placeholder/dialog."""
        if not self.current_user:
            return

        session_id = self.data_storage.create_session(self.current_user['id'], name)
        if session_id > 0:
            self.current_session = {'id': session_id, 'name': name}

            # Update sidebar
            self.sidebar.start_session(name)

            # Propagate
            self.dashboard_widget.set_session(session_id)
            self.live_analysis_widget.set_session(session_id)
            self.visualization_widget.set_session(session_id)
            self.replay_widget.set_session(session_id)

            # Update window title
            self.setWindowTitle(f"RSA v2 — {name}")

            # Switch to Live Analysis
            self.sidebar.set_active_index(1)

            return True
        return False

    def end_current_session(self):
        """Called when session is ended."""
        self.current_session = None
        self.sidebar.end_session()
        self.setWindowTitle("RSA v2 — Rifle Shooting Analysis")
        self.status_bar.showMessage("Session ended")

        # Reset widgets to placeholder state if they support it
        # (Widgets handle their own set_session(None) logic usually)

    def update_status_recording(self, is_recording):
        """Update window title and status based on recording state."""
        if not self.current_session:
            return

        name = self.current_session['name']
        if is_recording:
            self.setWindowTitle(f"RSA v2 — {name}  ●  Recording")
            self.analysis_status.setText(f"<span style='color:{theme.DANGER}'>●</span> Recording")
            self.analysis_status.setVisible(True)
            self.status_separator.setVisible(True)
        else:
            self.setWindowTitle(f"RSA v2 — {name}")
            self.analysis_status.setText(f"<span style='color:{theme.SUCCESS}'>●</span> Analysis Running")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.data_storage.close()
        event.accept()
