# Rifle Shooting Analysis Application

A desktop application for real-time and historical rifle shooting stability analysis, leveraging computer vision and fuzzy logic to provide feedback on shooting performance.

## Features

- **Real-Time Joint Tracking**: Uses MediaPipe Pose to track key joints (shoulders, elbows, wrists, hips, nose, ankles) in 3D space.
- **Stability Metrics Calculation**: Computes sway velocity, postural stability, and follow-through scores.
- **Shot Data Capture**: Automatically detects shots via audio threshold or manual triggering.
- **Historical Analysis**: Maintains baseline metrics and allows comparison of performance over time.
- **Real-Time Fuzzy Logic Feedback**: Provides actionable feedback based on stability metrics.
- **Interactive Visualizations**: Includes 3D visualization of joint positions and movement.
- **Session Recording & Replay**: Records and replays shooting sessions with analysis overlays.

## System Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - PyQt6 (UI framework)
  - MediaPipe (pose estimation)
  - OpenCV (computer vision)
  - NumPy (numerical operations)
  - Matplotlib (data visualization)
  - scikit-fuzzy (fuzzy logic system)
  - SQLite (data storage)
  - PyAudio (audio detection)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/rifle-shooting-analysis.git
   cd rifle-shooting-analysis
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python main.py
   ```

## Usage

1. **Registration/Login**:
   - Create a user account or log in with existing credentials.

2. **Create a Session**:
   - Click "New Session" to start a new shooting analysis session.

3. **Live Analysis**:
   - Go to the "Live Analysis" tab.
   - Click "Start Analysis" to begin tracking.
   - Position yourself in view of the camera.
   - The application will track your body positions and provide real-time feedback.
   - When you take a shot (either detected by sound or manually triggered), you can enter a subjective score.

4. **Review Dashboard**:
   - The "Dashboard" tab shows statistics and trends across all your sessions.
   - Select different sessions from the dropdown to view their statistics.

5. **3D Visualization**:
   - The "3D Visualization" tab allows you to see shot data in three dimensions.
   - Compare multiple shots to analyze differences in posture and stability.

6. **Session Replay**:
   - Record your sessions for later analysis.
   - Add overlays such as skeleton tracking and stability heatmaps.

7. **Settings**:
   - Customize camera/audio settings, algorithm parameters, and user interface preferences.

## Technical Documentation

The application is structured into several key modules:

- **Joint Tracking**: Utilizes MediaPipe Pose to extract 3D joint coordinates at 30+ FPS.
- **Stability Metrics**: Calculates sway velocity, postural stability, and follow-through scores.
- **Data Storage**: Manages user profiles, sessions, shots, and metrics in a SQLite database.
- **Fuzzy Logic Feedback**: Provides real-time feedback based on fuzzy logic rules.
- **User Interface**: Implements a modular interface with dedicated components for different functions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the pose estimation framework
- [PyQt](https://riverbankcomputing.com/software/pyqt/) for the UI framework
- [scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/) for the fuzzy logic implementation