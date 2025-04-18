import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, List, Optional

class FuzzyFeedback:
    """
    Implements fuzzy logic feedback system for rifle shooting analysis.
    Provides real-time feedback based on stability metrics.
    """
    
    def __init__(self):
        """Initialize the fuzzy feedback system with membership functions and rules."""
        # Create fuzzy variables
        self.setup_fuzzy_system()
        
        # Feedback message templates
        self.feedback_templates = {
            'wrist_stability': [
                "Stabilize your wrist.",
                "Keep your wrist steady.",
                "Focus on wrist support."
            ],
            'elbow_stability': [
                "Control your elbow movement.",
                "Keep your support arm steady.",
                "Minimize elbow motion."
            ],
            'stance': [
                "Center your stance.",
                "Balance your weight evenly.",
                "Maintain a stable base."
            ],
            'head_position': [
                "Keep your head steady.",
                "Maintain consistent cheek weld.",
                "Minimize head movement."
            ],
            'follow_through': [
                "Excellent follow-through.",
                "Maintain position after trigger break.",
                "Good stability through shot."
            ],
            'general_posture': [
                "Maintain consistent posture.",
                "Keep your body aligned.",
                "Control your breathing and posture."
            ]
        }
    
    def setup_fuzzy_system(self):
        """Set up the fuzzy control system with variables, membership functions, and rules."""
        # Define fuzzy variables (universes of discourse)
        
        # Sway velocities for different joints (mm/s)
        wrist_sway = ctrl.Antecedent(np.arange(0, 21, 1), 'wrist_sway')
        elbow_sway = ctrl.Antecedent(np.arange(0, 21, 1), 'elbow_sway')
        nose_sway = ctrl.Antecedent(np.arange(0, 21, 1), 'nose_sway')
        
        # Postural stability (px)
        hip_dev_x = ctrl.Antecedent(np.arange(0, 31, 1), 'hip_dev_x')
        nose_dev_y = ctrl.Antecedent(np.arange(0, 31, 1), 'nose_dev_y')
        
        # Follow-through score (0-1)
        follow_through = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'follow_through')
        
        # Output variable: feedback score (0-100)
        feedback_score = ctrl.Consequent(np.arange(0, 101, 1), 'feedback_score')
        
        # Define membership functions for inputs
        
        # Sway velocities
        wrist_sway['low'] = fuzz.trapmf(wrist_sway.universe, [0, 0, 3, 6])
        wrist_sway['medium'] = fuzz.trimf(wrist_sway.universe, [4, 7, 11])
        wrist_sway['high'] = fuzz.trapmf(wrist_sway.universe, [9, 12, 20, 20])
        
        elbow_sway['low'] = fuzz.trapmf(elbow_sway.universe, [0, 0, 2, 5])
        elbow_sway['medium'] = fuzz.trimf(elbow_sway.universe, [3, 6, 10])
        elbow_sway['high'] = fuzz.trapmf(elbow_sway.universe, [8, 11, 20, 20])
        
        nose_sway['low'] = fuzz.trapmf(nose_sway.universe, [0, 0, 1, 3])
        nose_sway['medium'] = fuzz.trimf(nose_sway.universe, [2, 4, 7])
        nose_sway['high'] = fuzz.trapmf(nose_sway.universe, [6, 8, 20, 20])
        
        # Postural stability
        hip_dev_x['low'] = fuzz.trapmf(hip_dev_x.universe, [0, 0, 5, 10])
        hip_dev_x['medium'] = fuzz.trimf(hip_dev_x.universe, [7, 12, 18])
        hip_dev_x['high'] = fuzz.trapmf(hip_dev_x.universe, [15, 20, 30, 30])
        
        nose_dev_y['low'] = fuzz.trapmf(nose_dev_y.universe, [0, 0, 3, 7])
        nose_dev_y['medium'] = fuzz.trimf(nose_dev_y.universe, [5, 9, 14])
        nose_dev_y['high'] = fuzz.trapmf(nose_dev_y.universe, [12, 16, 30, 30])
        
        # Follow-through score
        follow_through['poor'] = fuzz.trapmf(follow_through.universe, [0, 0, 0.3, 0.5])
        follow_through['average'] = fuzz.trimf(follow_through.universe, [0.4, 0.6, 0.8])
        follow_through['excellent'] = fuzz.trapmf(follow_through.universe, [0.7, 0.85, 1, 1])
        
        # Define membership functions for output
        feedback_score['poor'] = fuzz.trapmf(feedback_score.universe, [0, 0, 30, 45])
        feedback_score['average'] = fuzz.trimf(feedback_score.universe, [35, 50, 70])
        feedback_score['good'] = fuzz.trimf(feedback_score.universe, [60, 75, 90])
        feedback_score['excellent'] = fuzz.trapmf(feedback_score.universe, [80, 90, 100, 100])
        
        # Define fuzzy rules
        
        # Rule 1: Wrist and elbow stability
        rule1 = ctrl.Rule(
            wrist_sway['high'] | elbow_sway['high'],
            feedback_score['poor']
        )
        
        # Rule 2: Stance stability
        rule2 = ctrl.Rule(
            hip_dev_x['medium'] | hip_dev_x['high'],
            feedback_score['average']
        )
        
        # Rule 3: Head position
        rule3 = ctrl.Rule(
            nose_dev_y['medium'] | nose_dev_y['high'] | nose_sway['high'],
            feedback_score['average']
        )
        
        # Rule 4: Good follow-through
        rule4 = ctrl.Rule(
            follow_through['excellent'] & nose_sway['low'] & wrist_sway['low'],
            feedback_score['excellent']
        )
        
        # Rule 5: Decent stability
        rule5 = ctrl.Rule(
            (wrist_sway['medium'] & elbow_sway['medium']) |
            (nose_dev_y['medium'] & hip_dev_x['medium']),
            feedback_score['average']
        )
        
        # Rule 6: Overall good stability
        rule6 = ctrl.Rule(
            (wrist_sway['low'] & elbow_sway['low'] & nose_sway['medium']) |
            (follow_through['average'] & hip_dev_x['low']),
            feedback_score['good']
        )
        
        # Rule 7: Excellent stability
        rule7 = ctrl.Rule(
            (wrist_sway['low'] & elbow_sway['low'] & nose_sway['low'] & 
             hip_dev_x['low'] & nose_dev_y['low'] & follow_through['excellent']),
            feedback_score['excellent']
        )
        
        # Create control system
        self.feedback_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7
        ])
        
        # Create simulator
        self.feedback_simulator = ctrl.ControlSystemSimulation(self.feedback_ctrl)
    
    def generate_feedback(self, metrics: Dict) -> Dict:
        """
        Generate feedback based on stability metrics.
        
        Args:
            metrics: Dictionary containing stability metrics
            
        Returns:
            Dictionary with feedback score and text
        """
        # Extract metrics
        try:
            # Sway velocity for joints
            wrist_sway = (metrics.get('sway_velocity', {}).get('LEFT_WRIST', 0) +
                          metrics.get('sway_velocity', {}).get('RIGHT_WRIST', 0)) / 2
            
            elbow_sway = (metrics.get('sway_velocity', {}).get('LEFT_ELBOW', 0) +
                           metrics.get('sway_velocity', {}).get('RIGHT_ELBOW', 0)) / 2
            
            nose_sway = metrics.get('sway_velocity', {}).get('NOSE', 0)
            
            # Postural stability
            hip_dev_x = metrics.get('dev_x', {}).get('HIPS', 0)
            nose_dev_y = metrics.get('dev_y', {}).get('NOSE', 0)
            
            # Follow-through score
            follow_through = metrics.get('follow_through_score', 0.5)
            
            # Input values to fuzzy system
            self.feedback_simulator.input['wrist_sway'] = min(wrist_sway, 20)  # Cap at max value
            self.feedback_simulator.input['elbow_sway'] = min(elbow_sway, 20)
            self.feedback_simulator.input['nose_sway'] = min(nose_sway, 20)
            self.feedback_simulator.input['hip_dev_x'] = min(hip_dev_x, 30)
            self.feedback_simulator.input['nose_dev_y'] = min(nose_dev_y, 30)
            self.feedback_simulator.input['follow_through'] = max(0, min(follow_through, 1))  # Ensure in [0,1]
            
            # Compute result
            self.feedback_simulator.compute()
            
            # Get defuzzified result
            score = self.feedback_simulator.output['feedback_score']
            
            # Generate text feedback
            feedback_text = self._generate_text_feedback(metrics)
            
            return {
                'score': score,
                'text': feedback_text
            }
            
        except Exception as e:
            # Return default feedback if there's an error
            return {
                'score': 50,
                'text': "Keep your body stable and maintain consistent posture.",
                'error': str(e)
            }
    
    def _generate_text_feedback(self, metrics: Dict) -> str:
        """
        Generate text feedback based on metrics and fuzzy rules.
        
        Args:
            metrics: Dictionary containing stability metrics
            
        Returns:
            Text feedback string
        """
        feedback_items = []
        
        # Check wrist stability
        wrist_sway = (metrics.get('sway_velocity', {}).get('LEFT_WRIST', 0) +
                      metrics.get('sway_velocity', {}).get('RIGHT_WRIST', 0)) / 2
        if wrist_sway > 9:  # High wrist sway
            feedback_items.append(np.random.choice(self.feedback_templates['wrist_stability']))
        
        # Check elbow stability
        elbow_sway = (metrics.get('sway_velocity', {}).get('LEFT_ELBOW', 0) +
                      metrics.get('sway_velocity', {}).get('RIGHT_ELBOW', 0)) / 2
        if elbow_sway > 8:  # High elbow sway
            feedback_items.append(np.random.choice(self.feedback_templates['elbow_stability']))
        
        # Check stance (hip deviation)
        hip_dev_x = metrics.get('dev_x', {}).get('HIPS', 0)
        if hip_dev_x > 7:  # Medium or high hip deviation
            feedback_items.append(np.random.choice(self.feedback_templates['stance']))
        
        # Check head position
        nose_dev_y = metrics.get('dev_y', {}).get('NOSE', 0)
        nose_sway = metrics.get('sway_velocity', {}).get('NOSE', 0)
        if nose_dev_y > 5 or nose_sway > 6:  # Medium or high nose deviation/sway
            feedback_items.append(np.random.choice(self.feedback_templates['head_position']))
        
        # Check follow-through
        follow_through = metrics.get('follow_through_score', 0)
        if follow_through > 0.7 and wrist_sway < 6 and nose_sway < 3:  # Excellent follow-through
            feedback_items.append(np.random.choice(self.feedback_templates['follow_through']))
        
        # If no specific issues, give general feedback
        if not feedback_items:
            feedback_items.append(np.random.choice(self.feedback_templates['general_posture']))
        
        # Return combined feedback, limited to 2 items to keep it concise
        if len(feedback_items) > 2:
            feedback_items = feedback_items[:2]
        
        return ' '.join(feedback_items)
    
    def update_membership_functions(self, config: Dict):
        """
        Update membership function parameters based on configuration.
        
        Args:
            config: Dictionary of configuration values for membership functions
        """
        # This would allow customizing the fuzzy system parameters
        # For example, changing the thresholds for "low", "medium", "high" categories
        # Implementation would depend on specific requirements for customization
        pass