#!/usr/bin/env python3
"""
Rifle Shooting Analysis Desktop Application

A desktop application for macOS using VS Code, providing precise real-time and historical
rifle shooting stability analysis. Implements joint tracking, stability metrics calculation,
data storage, feedback, and user interface.
"""

import sys
import os
import logging
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QIcon

from src.ui.main_window import MainWindow

# Set up logging
def setup_logging():
    """Set up application logging."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'rifle_shot_analysis.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Rifle Shooting Analysis application")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Rifle Shooting Analysis")
    app.setOrganizationName("RifleShootingAnalysis")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create splash screen
    splash_path = os.path.join('resources', 'splash.png')
    if os.path.exists(splash_path):
        splash_pixmap = QPixmap(splash_path)
        splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        
        # Display splash screen for 2 seconds
        splash_timer = QTimer()
        splash_timer.singleShot(2000, splash.close)
    else:
        logger.warning("Splash screen image not found")
        splash = None
    
    # Create main window
    window = MainWindow()
    window.setWindowTitle("Rifle Shooting Analysis")
    
    # Set window icon
    icon_path = os.path.join('resources', 'icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        logger.warning("Application icon not found")
    
    # Show window
    if splash:
        # Wait for splash to close
        splash_timer.singleShot(2000, window.show)
    else:
        window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()