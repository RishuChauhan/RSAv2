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
        """Create professionally styled authentication widget with login and registration."""
        auth_widget = QWidget()
        layout = QVBoxLayout()
        
        # Add logo placeholder (would be replaced with actual logo in production)
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("margin: 20px;")
        # In a real app, you would use: logo_label.setPixmap(QPixmap("path/to/logo.png"))
        logo_label.setText("RIFLE SHOT ANALYSIS")
        logo_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #1976D2; margin: 30px; letter-spacing: 2px;")
        layout.addWidget(logo_label)
        
        # Professional subtitle
        subtitle = QLabel("Professional Shooting Performance Analysis")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 16px; color: #455A64; margin-bottom: 30px;")
        layout.addWidget(subtitle)
        
        # Create card-like container for auth forms
        auth_container = QWidget()
        auth_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #ECEFF1;
        """)
        auth_layout = QVBoxLayout()
        auth_container.setLayout(auth_layout)
        
        # Create tab widget for login/register with professional styling
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabBar::tab {
                padding: 10px 20px;
                font-size: 14px;
            }
        """)
        
        # Login tab
        login_widget = QWidget()
        login_layout = QVBoxLayout()
        
        login_title = QLabel("Sign In")
        login_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2; margin-bottom: 15px;")
        login_layout.addWidget(login_title)
        
        login_email = QLineEdit()
        login_email.setPlaceholderText("Email")
        login_email.setMinimumHeight(40)
        login_layout.addWidget(login_email)
        
        login_password = QLineEdit()
        login_password.setPlaceholderText("Password")
        login_password.setEchoMode(QLineEdit.EchoMode.Password)
        login_password.setMinimumHeight(40)
        login_layout.addWidget(login_password)
        
        login_button = QPushButton("Login")
        login_button.clicked.connect(lambda: self._handle_login(login_email.text(), login_password.text()))
        login_button.setMinimumHeight(40)
        login_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        login_layout.addWidget(login_button)
        
        login_widget.setLayout(login_layout)
        tab_widget.addTab(login_widget, "Login")
        
        # Register tab
        register_widget = QWidget()
        register_layout = QVBoxLayout()
        
        register_title = QLabel("Create Account")
        register_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2; margin-bottom: 15px;")
        register_layout.addWidget(register_title)
        
        register_name = QLineEdit()
        register_name.setPlaceholderText("Full Name")
        register_name.setMinimumHeight(40)
        register_layout.addWidget(register_name)
        
        register_email = QLineEdit()
        register_email.setPlaceholderText("Email")
        register_email.setMinimumHeight(40)
        register_layout.addWidget(register_email)
        
        register_password = QLineEdit()
        register_password.setPlaceholderText("Password")
        register_password.setEchoMode(QLineEdit.EchoMode.Password)
        register_password.setMinimumHeight(40)
        register_layout.addWidget(register_password)
        
        register_button = QPushButton("Register")
        register_button.clicked.connect(
            lambda: self._handle_register(
                register_name.text(), 
                register_email.text(), 
                register_password.text()
            )
        )
        register_button.setMinimumHeight(40)
        register_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        register_layout.addWidget(register_button)
        
        register_widget.setLayout(register_layout)
        tab_widget.addTab(register_widget, "Register")
        
        auth_layout.addWidget(tab_widget)
        layout.addWidget(auth_container)
        
        # Add some professional info text
        info_label = QLabel("Advanced analytics for precision shooting professionals")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: #757575; margin-top: 30px;")
        layout.addWidget(info_label)
        
        # Add version info
        version_label = QLabel("v1.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        version_label.setStyleSheet("color: #9E9E9E; margin: 10px;")
        layout.addWidget(version_label)
        
        # Add spacer at the bottom
        layout.addStretch()
        
        auth_widget.setLayout(layout)
        return auth_widget

    def _create_content_widget(self) -> QWidget:
        """Create the main content widget with professional styling and improved session flow."""
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create toolbar with professional styling
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #CFD8DC;
                spacing: 10px;
                padding: 5px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: #E3F2FD;
            }
            QToolButton:pressed {
                background-color: #BBDEFB;
            }
        """)
        
        # Add user info to toolbar
        self.user_info_label = QLabel("Welcome")
        self.user_info_label.setStyleSheet("font-weight: bold; color: #1976D2; padding: 0 10px;")
        toolbar.addWidget(self.user_info_label)
        
        toolbar.addSeparator()
        
        # Add session controls to toolbar with better styling
        self.session_label = QLabel("No active session")
        self.session_label.setStyleSheet("color: #455A64; padding: 0 10px; background-color: #F5F5F5; border-radius: 4px;")
        toolbar.addWidget(self.session_label)
        
        toolbar.addSeparator()
        
        # Create a styled button for session management (New Session/End Session)
        self.session_button = QPushButton("New Session")
        self.session_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.session_button.clicked.connect(self._handle_session_button)
        toolbar.addWidget(self.session_button)
        
        toolbar.addSeparator()
        
        # Create a styled logout button
        logout_action = QAction("Logout", self)
        logout_action.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogCloseButton))
        logout_action.triggered.connect(self._handle_logout)
        toolbar.addAction(logout_action)
        
        layout.addWidget(toolbar)
        
        # Create tab widget for different screens with professional styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CFD8DC;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                min-width: 120px;
                padding: 10px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:first {
                margin-left: 5px;
            }
        """)
        
        # Create and add the main tabs with professional icons
        self.dashboard_widget = DashboardWidget(self.data_storage)
        self.tab_widget.addTab(self.dashboard_widget, self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon), "Dashboard")
        
        self.live_analysis_widget = LiveAnalysisWidget(self.data_storage)
        self.tab_widget.addTab(self.live_analysis_widget, self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay), "Live Analysis")
        
        self.visualization_widget = VisualizationWidget(self.data_storage)
        self.tab_widget.addTab(self.visualization_widget, self.style().standardIcon(self.style().StandardPixmap.SP_ToolBarHorizontalExtensionButton), "3D Visualization")
        
        self.replay_widget = ReplayWidget(self.data_storage)
        self.tab_widget.addTab(self.replay_widget, self.style().standardIcon(self.style().StandardPixmap.SP_MediaSeekForward), "Session Replay")
        
        self.settings_widget = SettingsWidget(self.data_storage)
        self.tab_widget.addTab(self.settings_widget, self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogDetailedView), "Settings")
        
        layout.addWidget(self.tab_widget)
        
        # Create a professional status bar
        status_bar = QStatusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #ECEFF1;
                color: #455A64;
                border-top: 1px solid #CFD8DC;
            }
        """)
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        layout.addWidget(status_bar)
        
        content_widget.setLayout(layout)
        return content_widget
    
    def _apply_styling(self):
        """Apply CSS styling to the application."""
        # Professional blue color scheme
        style = """
        QMainWindow {
            background-color: #f5f8fa;
        }
        
        QLabel {
            font-size: 14px;
            color: #2c3e50;
        }
        
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 4px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #1976D2;
        }
        
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        
        QPushButton:disabled {
            background-color: #B0BEC5;
            color: #ECEFF1;
        }
        
        QLineEdit {
            padding: 8px;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            font-size: 14px;
            background-color: white;
        }
        
        QTabWidget::pane {
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #ECEFF1;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            color: #607D8B;
        }
        
        QTabBar::tab:selected {
            background-color: #2196F3;
            color: white;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            margin-top: 1.5ex;
            padding-top: 1ex;
            background-color: #FAFAFA;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
            color: #1976D2;
        }
        
        QComboBox {
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            padding: 6px 8px;
            min-width: 6em;
            background-color: white;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: #CFD8DC;
            border-left-style: solid;
        }
        
        QSpinBox, QDoubleSpinBox {
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
        }
        
        QProgressBar {
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
            color: white;
        }
        
        QProgressBar::chunk {
            background-color: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #E53935,
                stop:0.5 #FFB300,
                stop:1 #43A047
            );
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #CFD8DC;
            height: 8px;
            background: #ECEFF1;
            margin: 2px 0;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: #2196F3;
            border: 1px solid #1976D2;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        
        QTableWidget {
            alternate-background-color: #ECEFF1;
            background-color: white;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
        }
        
        QTableWidget::item:selected {
            background-color: #BBDEFB;
            color: #212121;
        }
        
        QHeaderView::section {
            background-color: #2196F3;
            color: white;
            padding: 5px;
            border: 1px solid #1976D2;
        }
        
        QStatusBar {
            background-color: #ECEFF1;
            color: #455A64;
        }
        
        QToolBar {
            background-color: #FFFFFF;
            border-bottom: 1px solid #CFD8DC;
            spacing: 5px;
        }
        
        QSplitter::handle {
            background-color: #CFD8DC;
        }
        
        QListWidget {
            background-color: white;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
        }
        
        QListWidget::item:selected {
            background-color: #BBDEFB;
            color: #212121;
        }
        
        QMenuBar {
            background-color: #FAFAFA;
        }
        
        QMenuBar::item:selected {
            background-color: #E3F2FD;
        }
        
        QMenu {
            background-color: white;
            border: 1px solid #CFD8DC;
        }
        
        QMenu::item:selected {
            background-color: #E3F2FD;
        }
        
        QCheckBox {
            spacing: 5px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        
        QCheckBox::indicator:unchecked {
            border: 1px solid #90A4AE;
            background-color: white;
            border-radius: 3px;
        }
        
        QCheckBox::indicator:checked {
            border: 1px solid #1976D2;
            background-color: #2196F3;
            border-radius: 3px;
        }
        """
        self.setStyleSheet(style)
    
    def _handle_login(self, email: str, password: str):
        """Handle user login with improved UX."""
        if not email or not password:
            QMessageBox.warning(self, "Login Error", "Please enter email and password.")
            return
        
        # Show a professional loading indicator
        self.statusBar().showMessage("Authenticating...")
        
        # Hash password for security
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Authenticate user
        user = self.data_storage.authenticate_user(email, password_hash)
        
        if user:
            self.current_user = user
            self.statusBar().showMessage(f"Logged in as {user['name']}")
            
            # Update user info in toolbar
            self.user_info_label.setText(f"Welcome, {user['name']}")
            
            # Switch to main content with a professional fade effect
            self.stacked_widget.setCurrentIndex(1)
            
            # Update all widgets with user data - make sure ALL components get the data
            self.dashboard_widget.set_user(user['id'])
            self.live_analysis_widget.set_user(user['id'])
            self.visualization_widget.set_user(user['id'])
            self.replay_widget.set_user(user['id'])
            self.settings_widget.set_user(user['id'])
            
            # If user had a previous session, reload it
            self.cursor = self.data_storage.conn.cursor()
            self.cursor.execute(
                """SELECT s.id, s.name FROM sessions s 
                WHERE s.user_id = ? 
                ORDER BY s.created_at DESC LIMIT 1""", 
                (user['id'],)
            )
            last_session = self.cursor.fetchone()
            
            if last_session:
                self.current_session = {
                    'id': last_session['id'],
                    'name': last_session['name']
                }
                self.session_label.setText(f"Active Session: {last_session['name']}")
                
                # Update all views with the last session
                self.dashboard_widget.set_session(last_session['id'])
                self.live_analysis_widget.set_session(last_session['id'])
                self.visualization_widget.set_session(last_session['id'])
                self.replay_widget.set_session(last_session['id'])
            
            # Show a welcome message
            QMessageBox.information(self, "Welcome", f"Welcome {user['name']}!\n\nYou are now logged into the Rifle Shooting Analysis professional system.")
        else:
            self.statusBar().showMessage("Login failed")
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
    
    def _handle_session_button(self):
        """Handle session button click (New Session/End Session)."""
        if not hasattr(self, 'current_session') or self.current_session is None:
            self._create_new_session()
        else:
            self._end_session()

    def _create_new_session(self):
        """Create a new shooting session with professional UX."""
        if not self.current_user:
            return
        
        # Use a more professional dialog design
        from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("New Session")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        # Session name field
        from PyQt6.QtWidgets import QLineEdit
        session_name = QLineEdit()
        session_name.setPlaceholderText("Enter a descriptive name for this session")
        session_name.setMinimumHeight(30)
        layout.addRow("Session Name:", session_name)
        
        # Optional description field
        session_desc = QLineEdit()
        session_desc.setPlaceholderText("Optional description")
        session_desc.setMinimumHeight(30)
        layout.addRow("Description:", session_desc)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        # Execute dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = session_name.text().strip()
            if name:
                self.statusBar().showMessage("Creating new session...")
                
                session_id = self.data_storage.create_session(
                    self.current_user['id'], name
                )
                
                if session_id > 0:
                    self.current_session = {
                        'id': session_id,
                        'name': name
                    }
                    
                    self.session_label.setText(f"Active Session: {name}")
                    self.statusBar().showMessage(f"Created new session: {name}")
                    self.session_button.setText("End Session")
                    
                    # Update widgets with new session
                    self.dashboard_widget.set_user(self.current_user['id'])  # Refresh sessions in dashboard
                    self.dashboard_widget.set_session(session_id)
                    
                    self.live_analysis_widget.set_session(session_id)
                    
                    self.visualization_widget.set_user(self.current_user['id'])  # Refresh sessions
                    self.visualization_widget.set_session(session_id)
                    
                    self.replay_widget.set_user(self.current_user['id'])  # Refresh sessions
                    self.replay_widget.set_session(session_id)
                    
                    # Switch to live analysis tab and automatically start analysis
                    self.tab_widget.setCurrentIndex(1)  # Switch to Live Analysis tab
                    QTimer.singleShot(500, self.live_analysis_widget.start_analysis)  # Start analysis after a short delay
                    
                    # Show success message
                    QMessageBox.information(self, "Session Created", 
                                        f"Session '{name}' created successfully.\n\nLive analysis will start automatically.")
                else:
                    self.statusBar().showMessage("Failed to create session")
                    QMessageBox.critical(self, "Error", "Failed to create session.")
            else:
                QMessageBox.warning(self, "Invalid Name", "Please enter a valid session name.")
    
    def _end_session(self):
        """End the current session."""
        if not self.current_session:
            return
            
        # Ask for confirmation
        reply = QMessageBox.question(
            self, "End Session", 
            "Are you sure you want to end the current session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # End session in Live Analysis widget
        if hasattr(self.live_analysis_widget, 'end_session'):
            self.live_analysis_widget.end_session()
        
        # Reset session info
        self.current_session = None
        self.session_label.setText("No active session")
        self.session_button.setText("New Session")
        
        # Update status
        self.statusBar().showMessage("Session ended")

    
    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up resources
        self.data_storage.close()
        event.accept()