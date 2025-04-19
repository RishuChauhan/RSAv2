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
        
        Args:
            joint_history: List of joint data dictionaries with timestamps
            shot_time: Time of the shot/trigger pull
            post_window: Size of the post-shot window in seconds (default: 1.0s)
                
        Returns:
            Follow-through score between 0 and 1 (higher is better)
        """
        # Only keep frames from [shot_time, shot_time + post_window]
        post = [f for f in joint_history if shot_time <= f['timestamp'] <= shot_time + post_window]
        
        # Alternative using the _slice_window helper:
        # post = self._slice_window(joint_history, window_sec=post_window, end_time=shot_time + post_window)
        
        # Need at least 2 frames to calculate velocities and deviations
        if len(post) < 2:
            return 0.0
        
        # Calculate sway velocity on post-shot window
        sway_velocities = {}
        
        # Sort by timestamp to ensure correct calculation
        sorted_history = sorted(post, key=lambda x: x['timestamp'])
        
        # Get all unique joint names from the first entry
        if not sorted_history or 'joints' not in sorted_history[0]:
            return 0.0
            
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
        
        # Calculate postural stability on post-shot window
        # Extract X and Y coordinates for each joint
        joint_coords = {joint: {'x': [], 'y': []} for joint in joint_names}
        
        for data in sorted_history:
            for joint_name in joint_names:
                if joint_name in data['joints']:
                    joint = data['joints'][joint_name]
                    joint_coords[joint_name]['x'].append(joint['x'])
                    joint_coords[joint_name]['y'].append(joint['y'])
        
        # Calculate standard deviation for each joint
        dev_x = {}
        dev_y = {}
        
        for joint_name in joint_names:
            x_coords = joint_coords[joint_name]['x']
            y_coords = joint_coords[joint_name]['y']
            
            if x_coords and y_coords:
                dev_x[joint_name] = np.std(x_coords)
                dev_y[joint_name] = np.std(y_coords)
            else:
                dev_x[joint_name] = 0.0
                dev_y[joint_name] = 0.0
        
        # Calculate weighted sway sum
        w_sway = 0.0
        weights_sum = 0.0
        
        # Map individual joints to groups for weighting
        joint_to_group = {
            'LEFT_SHOULDER': 'SHOULDERS', 'RIGHT_SHOULDER': 'SHOULDERS',
            'LEFT_ELBOW': 'ELBOWS', 'RIGHT_ELBOW': 'ELBOWS',
            'LEFT_WRIST': 'WRISTS', 'RIGHT_WRIST': 'WRISTS',
            'NOSE': 'NOSE'
        }
        
        for joint, velocity in sway_velocities.items():
            if joint in joint_to_group:
                group = joint_to_group[joint]
                if group in self.joint_weights:
                    weight = self.joint_weights[group]
                    w_sway += weight * velocity
                    weights_sum += weight
        
        # Normalize if we have weights
        if weights_sum > 0:
            w_sway /= weights_sum
        
        # Calculate postural deviation penalty
        p_dev = 0.0
        
        # Calculate mean positions for reference
        mean_positions = {}
        for joint in joint_to_group.keys():
            x_coords = []
            y_coords = []
            
            for data in sorted_history:
                if 'joints' in data and joint in data['joints']:
                    x_coords.append(data['joints'][joint]['x'])
                    y_coords.append(data['joints'][joint]['y'])
            
            if x_coords and y_coords:
                mean_positions[joint] = {
                    'x': np.mean(x_coords),
                    'y': np.mean(y_coords)
                }
        
        # Calculate deviations from mean positions
        for data in sorted_history:
            if 'joints' not in data:
                continue
                
            for joint in joint_to_group.keys():
                if joint in data['joints'] and joint in mean_positions:
                    j_data = data['joints'][joint]
                    mean_pos = mean_positions[joint]
                    
                    # Absolute deviations
                    dev_x_val = abs(j_data['x'] - mean_pos['x'])
                    dev_y_val = abs(j_data['y'] - mean_pos['y'])
                    
                    # Add weighted deviations to penalty
                    p_dev += (self.postural_weights['x'] * dev_x_val + 
                            self.postural_weights['y'] * dev_y_val)
        
        # Normalize by number of data points and joints
        data_points = len(sorted_history)
        joint_count = len(joint_to_group)
        if data_points > 0 and joint_count > 0:
            p_dev /= (data_points * joint_count)
        
        # Compute combined score with negation to make it a "higher is better" scale
        u = -(self.lambda_sway * w_sway + self.lambda_dev * p_dev)
        
        # Compute follow-through score with classic sigmoid
        # F(t) = 1/(1+e^(-u))
        # Now higher u (better stability) → F closer to 1
        follow_through = 1.0 / (1.0 + math.exp(-u))
        
        return follow_through
    
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

