import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class StabilityMetrics:
    """
    Calculate stability metrics for rifle shooting analysis based on joint tracking data.
    Implements sway velocity, postural stability, and follow-through scores.
    """
    
    def __init__(self):
        """Initialize the stability metrics calculator."""
        self.baseline_metrics = {}
        
        # Joint weights for follow-through calculation
        self.joint_weights = {
            'SHOULDERS': 0.2,  # Average of LEFT and RIGHT
            'ELBOWS': 0.3,     # Average of LEFT and RIGHT
            'WRISTS': 0.4,     # Average of LEFT and RIGHT
            'NOSE': 0.1
        }
        
        # Postural deviation weights (β)
        self.postural_weights = {
            'x': 0.5,  # Horizontal deviation weight
            'y': 0.5   # Vertical deviation weight
        }
        
        # Follow-through score parameters (λ)
        self.lambda_sway = 0.5
        self.lambda_dev = 0.5
    
    def calculate_sway_velocity(self, joint_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate sway velocity for each tracked joint within 1.0 second window.
        
        Args:
            joint_history: List of joint data dictionaries with timestamps
                
        Returns:
            Dictionary of sway velocities for each joint
        """
        if len(joint_history) < 2:
            return {}
        
        # Slice to 1.0 second window before the shot
        sliced_history = self._slice_window(joint_history, 1.0)
        
        # If not enough frames in window, use all available frames
        if len(sliced_history) < 2:
            sliced_history = joint_history
        
        # Sort by timestamp to ensure correct calculation
        sorted_history = sorted(sliced_history, key=lambda x: x['timestamp'])
        
        # Initialize result dictionary
        sway_velocities = {}
        
        # Get all unique joint names from the first entry
        if not sorted_history or 'joints' not in sorted_history[0]:
            return {}
            
        joint_names = sorted_history[0]['joints'].keys()
        
        # Calculate velocities for each joint
        for joint_name in joint_names:
            velocities = []
            
            for i in range(1, len(sorted_history)):
                prev_data = sorted_history[i-1]
                curr_data = sorted_history[i]
                
                # Skip if joint data is missing
                if (joint_name not in prev_data['joints'] or 
                    joint_name not in curr_data['joints']):
                    continue
                
                prev_joint = prev_data['joints'][joint_name]
                curr_joint = curr_data['joints'][joint_name]
                
                # Time difference between samples
                dt = curr_data['timestamp'] - prev_data['timestamp']
                if dt <= 0:
                    continue  # Skip invalid time differences
                
                # Calculate Euclidean distance in 3D
                dx = curr_joint['x'] - prev_joint['x']
                dy = curr_joint['y'] - prev_joint['y']
                dz = curr_joint['z'] - prev_joint['z']
                
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Velocity = distance / time
                velocity = distance / dt
                velocities.append(velocity)
            
            # Calculate average velocity if we have data
            if velocities:
                avg_velocity = sum(velocities) / len(velocities)
                sway_velocities[joint_name] = avg_velocity
            else:
                sway_velocities[joint_name] = 0.0
        
        # Add aggregated joint group velocities that match UI expectations
        sway_velocities['SHOULDERS'] = (sway_velocities.get('LEFT_SHOULDER', 0) + 
                                    sway_velocities.get('RIGHT_SHOULDER', 0)) / 2
        
        sway_velocities['ELBOWS'] = (sway_velocities.get('LEFT_ELBOW', 0) + 
                                    sway_velocities.get('RIGHT_ELBOW', 0)) / 2
        
        sway_velocities['WRISTS'] = (sway_velocities.get('LEFT_WRIST', 0) + 
                                    sway_velocities.get('RIGHT_WRIST', 0)) / 2
        
        sway_velocities['HIPS'] = (sway_velocities.get('LEFT_HIP', 0) + 
                                sway_velocities.get('RIGHT_HIP', 0)) / 2
        
        sway_velocities['ANKLES'] = (sway_velocities.get('LEFT_ANKLE', 0) + 
                                    sway_velocities.get('RIGHT_ANKLE', 0)) / 2
        
        return sway_velocities

    def calculate_postural_stability(self, joint_history: List[Dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate postural stability (DevX, DevY) within 0.6 second window.
        
        Args:
            joint_history: List of joint data dictionaries with timestamps
                
        Returns:
            Tuple of dictionaries (DevX, DevY) for each joint and body part group
        """
        if not joint_history:
            return {}, {}
        
        # Slice to 0.6 second window before the shot
        sliced_history = self._slice_window(joint_history, 0.6)
        
        # If not enough frames in window, use all available frames
        if len(sliced_history) < 2:
            sliced_history = joint_history
                
        # Initialize result dictionaries
        dev_x = {}
        dev_y = {}
        
        # Get all unique joint names from the first entry
        if 'joints' not in sliced_history[0]:
            return {}, {}
                
        joint_names = sliced_history[0]['joints'].keys()
        
        # Extract X and Y coordinates for each joint
        joint_coords = {joint: {'x': [], 'y': []} for joint in joint_names}
        
        for data in sliced_history:
            for joint_name in joint_names:
                if joint_name in data['joints']:
                    joint = data['joints'][joint_name]
                    joint_coords[joint_name]['x'].append(joint['x'])
                    joint_coords[joint_name]['y'].append(joint['y'])
        
        # Calculate standard deviation for each joint
        for joint_name in joint_names:
            x_coords = joint_coords[joint_name]['x']
            y_coords = joint_coords[joint_name]['y']
            
            if x_coords and y_coords:
                dev_x[joint_name] = np.std(x_coords)
                dev_y[joint_name] = np.std(y_coords)
            else:
                dev_x[joint_name] = 0.0
                dev_y[joint_name] = 0.0
        
        # Calculate aggregate DevX and DevY for body part groups
        # Upper body
        dev_x['SHOULDERS'] = (dev_x.get('LEFT_SHOULDER', 0) + dev_x.get('RIGHT_SHOULDER', 0)) / 2
        dev_y['SHOULDERS'] = (dev_y.get('LEFT_SHOULDER', 0) + dev_y.get('RIGHT_SHOULDER', 0)) / 2
        
        dev_x['ELBOWS'] = (dev_x.get('LEFT_ELBOW', 0) + dev_x.get('RIGHT_ELBOW', 0)) / 2
        dev_y['ELBOWS'] = (dev_y.get('LEFT_ELBOW', 0) + dev_y.get('RIGHT_ELBOW', 0)) / 2
        
        dev_x['WRISTS'] = (dev_x.get('LEFT_WRIST', 0) + dev_x.get('RIGHT_WRIST', 0)) / 2
        dev_y['WRISTS'] = (dev_y.get('LEFT_WRIST', 0) + dev_y.get('RIGHT_WRIST', 0)) / 2
        
        # Lower body
        dev_x['HIPS'] = (dev_x.get('LEFT_HIP', 0) + dev_x.get('RIGHT_HIP', 0)) / 2
        dev_y['HIPS'] = (dev_y.get('LEFT_HIP', 0) + dev_y.get('RIGHT_HIP', 0)) / 2
        
        dev_x['ANKLES'] = (dev_x.get('LEFT_ANKLE', 0) + dev_x.get('RIGHT_ANKLE', 0)) / 2
        dev_y['ANKLES'] = (dev_y.get('LEFT_ANKLE', 0) + dev_y.get('RIGHT_ANKLE', 0)) / 2
        
        # Upper body aggregate
        upper_body_joints = ['SHOULDERS', 'ELBOWS', 'WRISTS', 'NOSE']
        upper_body_count = sum(1 for j in upper_body_joints if j in dev_x)
        
        if upper_body_count > 0:
            dev_x['UPPER_BODY'] = sum(dev_x.get(j, 0) for j in upper_body_joints) / upper_body_count
            dev_y['UPPER_BODY'] = sum(dev_y.get(j, 0) for j in upper_body_joints) / upper_body_count
        
        return dev_x, dev_y

    def calculate_follow_through_score(self, joint_history: List[Dict], shot_time: float, post_window: float = 1.0) -> float:
        """
        Calculate follow-through score based on stability in the 1-second window AFTER the shot.
        A robust implementation that ensures proper post-shot analysis.
        
        Args:
            joint_history: List of joint data dictionaries with timestamps
            shot_time: Time of the shot/trigger pull
            post_window: Size of the post-shot window in seconds (default: 1.0s)
                
        Returns:
            Follow-through score between 0 and 1 (higher is better)
        """
        # Validate inputs
        if not joint_history:
            print("Warning: Empty joint history provided")
            return 0.1
        
        # Validate shot_time - if not provided, use a reasonable default
        if shot_time is None:
            print("Warning: No shot time provided, using last frame timestamp - 1.0s")
            try:
                shot_time = joint_history[-1]['timestamp'] - 1.0
            except (KeyError, IndexError):
                print("Error: Cannot determine shot time from joint history")
                return 0.1
        
        # Extract only post-shot frames [shot_time, shot_time + post_window]
        post_shot_frames = []
        for frame in joint_history:
            try:
                frame_time = frame['timestamp']
                if shot_time <= frame_time <= shot_time + post_window:
                    post_shot_frames.append(frame)
            except (KeyError, TypeError) as e:
                print(f"Warning: Error extracting timestamp from frame: {e}")
                continue
        
        # Check if we have enough frames in post-shot window
        if len(post_shot_frames) < 2:
            print(f"Warning: Not enough frames in post-shot window ({len(post_shot_frames)} frames). Need at least 2.")
            # IMPORTANT: Return a low score instead of using the entire history
            # This ensures we don't falsely evaluate follow-through when we have insufficient data
            return 0.2
        
        # Sort frames by timestamp to ensure proper order
        post_shot_frames = sorted(post_shot_frames, key=lambda x: x.get('timestamp', 0))
        
        # ========== STABILITY CALCULATION ==========
        # Calculate joint stability in the post-shot window using a simplified approach
        
        # Track movement for key joints (upper body joints most relevant for shooting)
        key_joint_groups = {
            'SHOULDERS': ['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
            'ELBOWS': ['LEFT_ELBOW', 'RIGHT_ELBOW'],
            'WRISTS': ['LEFT_WRIST', 'RIGHT_WRIST'],
            'NOSE': ['NOSE']
        }
        
        # Joint importance weights
        joint_weights = {
            'SHOULDERS': 0.2,
            'ELBOWS': 0.3,
            'WRISTS': 0.4,
            'NOSE': 0.1
        }
        
        # Calculate average movement for each joint group
        joint_movement = {group: 0.0 for group in key_joint_groups}
        joint_counts = {group: 0 for group in key_joint_groups}
        
        # Process each frame pair to calculate movement
        for i in range(1, len(post_shot_frames)):
            curr_frame = post_shot_frames[i]
            prev_frame = post_shot_frames[i-1]
            
            # Calculate time difference
            try:
                dt = curr_frame['timestamp'] - prev_frame['timestamp']
                if dt <= 0:
                    continue  # Skip invalid time differences
            except (KeyError, TypeError):
                continue  # Skip frames with invalid timestamps
            
            # Check all joint groups
            for group, joints in key_joint_groups.items():
                group_movement = 0.0
                group_count = 0
                
                # Calculate movement for each joint in the group
                for joint_name in joints:
                    try:
                        # Extract previous and current positions
                        if ('joints' not in prev_frame or joint_name not in prev_frame['joints'] or
                            'joints' not in curr_frame or joint_name not in curr_frame['joints']):
                            continue
                        
                        prev_joint = prev_frame['joints'][joint_name]
                        curr_joint = curr_frame['joints'][joint_name]
                        
                        # Check for required fields
                        if not all(k in prev_joint for k in ['x', 'y', 'z']) or not all(k in curr_joint for k in ['x', 'y', 'z']):
                            continue
                        
                        # Calculate Euclidean distance in 3D
                        dx = curr_joint['x'] - prev_joint['x']
                        dy = curr_joint['y'] - prev_joint['y']
                        dz = curr_joint['z'] - prev_joint['z']
                        
                        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                        
                        # Movement rate (distance/time)
                        movement_rate = distance / dt
                        
                        group_movement += movement_rate
                        group_count += 1
                    except (KeyError, TypeError, ZeroDivisionError) as e:
                        print(f"Warning: Error calculating movement for {joint_name}: {e}")
                        continue
                
                # Add the average movement for this group
                if group_count > 0:
                    joint_movement[group] += group_movement / group_count
                    joint_counts[group] += 1
        
        # Calculate average movement per group
        avg_movements = {}
        for group in key_joint_groups:
            if joint_counts[group] > 0:
                avg_movements[group] = joint_movement[group] / joint_counts[group]
            else:
                avg_movements[group] = 0.0
        
        # Calculate weighted stability score (inverse of movement - less movement is more stable)
        weighted_score = 0.0
        total_weight = 0.0
        
        for group, movement in avg_movements.items():
            weight = joint_weights.get(group, 0.0)
            
            # Convert movement to stability (0-1 scale, smaller movement = higher stability)
            # Calibrated thresholds: 0 mm/s → 1.0 stability, 20+ mm/s → 0.0 stability
            stability = max(0.0, 1.0 - (movement / 20.0))
            
            weighted_score += weight * stability
            total_weight += weight
        
        # Calculate final follow-through score
        if total_weight > 0:
            follow_through_score = weighted_score / total_weight
        else:
            follow_through_score = 0.3  # Default score if weighting fails
        
        # Sanity check to ensure score is in valid range [0,1]
        follow_through_score = max(0.0, min(1.0, follow_through_score))
        
        return follow_through_score
    
    def compare_to_baseline(self, metrics: Dict, metric_type: str) -> Dict:
        """
        Compare current metrics to baseline best-shot metrics.
        
        Args:
            metrics: Current calculated metrics
            metric_type: Type of metric ('sway_velocity', 'dev_x', 'dev_y', 'follow_through')
            
        Returns:
            Dictionary of percentage differences relative to baseline
        """
        if metric_type not in self.baseline_metrics:
            return {}
        
        baseline = self.baseline_metrics[metric_type]
        comparison = {}
        
        # Calculate percentage differences
        for key, value in metrics.items():
            if key in baseline and baseline[key] > 0:
                # For sway velocity and deviation, lower is better so use negative percentage
                if metric_type in ['sway_velocity', 'dev_x', 'dev_y']:
                    comparison[key] = ((value - baseline[key]) / baseline[key]) * 100
                # For follow-through, higher is better so use positive percentage
                elif metric_type == 'follow_through':
                    comparison[key] = ((value - baseline[key]) / baseline[key]) * 100
            else:
                comparison[key] = 0.0
        
        return comparison
    
    def update_baseline(self, joint_history: List[Dict], subjective_score: float,
                       current_best_score: float) -> bool:
        """
        Update baseline metrics if the current subjective score is higher than the best.
        
        Args:
            joint_history: List of joint data dictionaries
            subjective_score: Current subjective score (1-10)
            current_best_score: Current best subjective score
            
        Returns:
            Boolean indicating if baseline was updated
        """
        if subjective_score <= current_best_score:
            return False
        
        # Calculate all metrics
        sway_velocities = self.calculate_sway_velocity(joint_history)
        dev_x, dev_y = self.calculate_postural_stability(joint_history)
        follow_through = self.calculate_follow_through_score(joint_history)
        
        # Update baseline
        self.baseline_metrics['sway_velocity'] = sway_velocities
        self.baseline_metrics['dev_x'] = dev_x
        self.baseline_metrics['dev_y'] = dev_y
        self.baseline_metrics['follow_through'] = follow_through
        
        return True
    
    def set_joint_weights(self, weights: Dict[str, float]):
        """Set joint weights for follow-through calculation."""
        self.joint_weights = weights
    
    def set_postural_weights(self, x_weight: float, y_weight: float):
        """Set postural deviation weights."""
        self.postural_weights = {'x': x_weight, 'y': y_weight}
    
    def set_lambda_parameters(self, lambda_sway: float, lambda_dev: float):
        """Set lambda parameters for follow-through calculation."""
        self.lambda_sway = lambda_sway
        self.lambda_dev = lambda_dev

    def _slice_window(self, history, window_sec, end_time=None):
        """
        Slice joint history to get only frames within the specified time window.
        
        Args:
            history: List of joint data dictionaries with timestamps
            window_sec: Window size in seconds
            end_time: Optional end time of the window. If None, uses the last frame's timestamp.
            
        Returns:
            Sliced history list containing only frames within window
        """
        if not history:
            return []
            
        # Get end timestamp (either provided or most recent frame)
        if end_time is None:
            t_end = history[-1]['timestamp']
        else:
            t_end = end_time
        
        # Calculate cutoff time
        cutoff = t_end - window_sec
        
        # Filter history for the desired window
        # For pre-shot windows: frames between [t_end - window_sec, t_end]
        # For post-shot windows: frames between [t_end - window_sec, t_end]
        return [f for f in history if cutoff <= f['timestamp'] <= t_end]