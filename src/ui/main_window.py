from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QStackedWidget, QMessageBox,
    QTabWidget, QLineEdit, QToolBar, QStatusBar
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QAction

import os
import hashlib
import sys

# Import UI components
from .dashboard import DashboardWidget
from .live_analysis import LiveAnalysisWidget
from .visualization import VisualizationWidget
from .replay import ReplayWidget
from .settings import SettingsWidget

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
        self.setWindowTitle("Rifle Shooting Analysis")
        self.setMinimumSize(1024, 768)
        
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create login/register widget
        self.auth_widget = self._create_auth_widget()
        self.stacked_widget.addWidget(self.auth_widget)
        
        # Create main content widget (shown after login)
        self.content_widget = self._create_content_widget()
        self.stacked_widget.addWidget(self.content_widget)
        
        # Start with authentication screen
        self.stacked_widget.setCurrentIndex(0)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to Rifle Shooting Analysis")
        
        # Apply styling
        self._apply_styling()
    
    def _create_auth_widget(self) -> QWidget:
        """Create authentication widget with login and registration."""
        auth_widget = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Rifle Shooting Analysis")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title_label)
        
        # Create tab widget for login/register
        tab_widget = QTabWidget()
        
        # Login tab
        login_widget = QWidget()
        login_layout = QVBoxLayout()
        
        login_email = QLineEdit()
        login_email.setPlaceholderText("Email")
        login_layout.addWidget(login_email)
        
        login_password = QLineEdit()
        login_password.setPlaceholderText("Password")
        login_password.setEchoMode(QLineEdit.EchoMode.Password)
        login_layout.addWidget(login_password)
        
        login_button = QPushButton("Login")
        login_button.clicked.connect(lambda: self._handle_login(login_email.text(), login_password.text()))
        login_layout.addWidget(login_button)
        
        login_widget.setLayout(login_layout)
        tab_widget.addTab(login_widget, "Login")
        
        # Register tab
        register_widget = QWidget()
        register_layout = QVBoxLayout()
        
        register_name = QLineEdit()
        register_name.setPlaceholderText("Full Name")
        register_layout.addWidget(register_name)
        
        register_email = QLineEdit()
        register_email.setPlaceholderText("Email")
        register_layout.addWidget(register_email)
        
        register_password = QLineEdit()
        register_password.setPlaceholderText("Password")
        register_password.setEchoMode(QLineEdit.EchoMode.Password)
        register_layout.addWidget(register_password)
        
        register_button = QPushButton("Register")
        register_button.clicked.connect(
            lambda: self._handle_register(
                register_name.text(), 
                register_email.text(), 
                register_password.text()
            )
        )
        register_layout.addWidget(register_button)
        
        register_widget.setLayout(register_layout)
        tab_widget.addTab(register_widget, "Register")
        
        layout.addWidget(tab_widget)
        
        # Add spacer at the bottom
        layout.addStretch()
        
        auth_widget.setLayout(layout)
        return auth_widget
    
    def _create_content_widget(self) -> QWidget:
        """Create the main content widget with tabs for different features."""
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(32, 32))
        
        # Add session controls to toolbar
        self.session_label = QLabel("No active session")
        toolbar.addWidget(self.session_label)
        
        toolbar.addSeparator()
        
        new_session_action = QAction("New Session", self)
        new_session_action.triggered.connect(self._create_new_session)
        toolbar.addAction(new_session_action)
        
        toolbar.addSeparator()
        
        logout_action = QAction("Logout", self)
        logout_action.triggered.connect(self._handle_logout)
        toolbar.addAction(logout_action)
        
        layout.addWidget(toolbar)
        
        # Create tab widget for different screens
        tab_widget = QTabWidget()
        
        # Create and add the main tabs
        self.dashboard_widget = DashboardWidget(self.data_storage)
        tab_widget.addTab(self.dashboard_widget, "Dashboard")
        
        self.live_analysis_widget = LiveAnalysisWidget(self.data_storage)
        tab_widget.addTab(self.live_analysis_widget, "Live Analysis")
        
        self.visualization_widget = VisualizationWidget(self.data_storage)
        tab_widget.addTab(self.visualization_widget, "3D Visualization")
        
        self.replay_widget = ReplayWidget(self.data_storage)
        tab_widget.addTab(self.replay_widget, "Session Replay")
        
        self.settings_widget = SettingsWidget(self.data_storage)
        tab_widget.addTab(self.settings_widget, "Settings")
        
        layout.addWidget(tab_widget)
        
        content_widget.setLayout(layout)
        return content_widget
    
    def _apply_styling(self):
        """Apply CSS styling to the application."""
        # Simple modern styling
        style = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        
        QLabel {
            font-size: 14px;
        }
        
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 4px;
        }
        
        QPushButton:hover {
            background-color: #45a049;
        }
        
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        
        QLineEdit {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        QTabWidget::pane {
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #4CAF50;
            color: white;
        }
        """
        self.setStyleSheet(style)
    
    def _handle_login(self, email: str, password: str):
        """Handle user login."""
        if not email or not password:
            QMessageBox.warning(self, "Login Error", "Please enter email and password.")
            return
        
        # Hash password for security
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Authenticate user
        user = self.data_storage.authenticate_user(email, password_hash)
        
        if user:
            self.current_user = user
            self.status_bar.showMessage(f"Logged in as {user['name']}")
            
            # Switch to main content
            self.stacked_widget.setCurrentIndex(1)
            
            # Update dashboard with user data
            self.dashboard_widget.set_user(user['id'])
            self.live_analysis_widget.set_user(user['id'])
            self.visualization_widget.set_user(user['id'])
            self.replay_widget.set_user(user['id'])
            self.settings_widget.set_user(user['id'])
        else:
            QMessageBox.critical(self, "Login Error", "Invalid email or password.")
    
    def _handle_register(self, name: str, email: str, password: str):
        """Handle user registration."""
        if not name or not email or not password:
            QMessageBox.warning(self, "Registration Error", "Please fill all fields.")
            return
        
        # Simple validation
        if '@' not in email or '.' not in email:
            QMessageBox.warning(self, "Registration Error", "Please enter a valid email.")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "Registration Error", "Password must be at least 6 characters.")
            return
        
        # Hash password for security
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Create user
        user_id = self.data_storage.create_user(name, email, password_hash)
        
        if user_id > 0:
            QMessageBox.information(self, "Registration Successful", 
                                   "Account created successfully. You can now log in.")
        else:
            QMessageBox.critical(self, "Registration Error", 
                                "Registration failed. Email may already be in use.")
    
    def _handle_logout(self):
        """Handle user logout."""
        self.current_user = None
        self.current_session = None
        
        # Reset widgets
        self.session_label.setText("No active session")
        
        # Switch back to login screen
        self.stacked_widget.setCurrentIndex(0)
        self.status_bar.showMessage("Logged out")
    
    def _create_new_session(self):
        """Create a new shooting session."""
        if not self.current_user:
            return
        
        # Simple dialog for session name
        from PyQt6.QtWidgets import QInputDialog
        
        session_name, ok = QInputDialog.getText(
            self, "New Session", "Enter session name:"
        )
        
        if ok and session_name:
            session_id = self.data_storage.create_session(
                self.current_user['id'], session_name
            )
            
            if session_id > 0:
                self.current_session = {
                    'id': session_id,
                    'name': session_name
                }
                
                self.session_label.setText(f"Active Session: {session_name}")
                self.status_bar.showMessage(f"Created new session: {session_name}")
                
                # Update widgets with new session
                self.live_analysis_widget.set_session(session_id)
                self.visualization_widget.set_session(session_id)
                self.replay_widget.set_session(session_id)
                
                # Switch to live analysis tab
                self.stacked_widget.widget(1).findChild(QTabWidget).setCurrentIndex(1)
            else:
                QMessageBox.critical(self, "Error", "Failed to create session.")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up resources
        self.data_storage.close()
        event.accept()