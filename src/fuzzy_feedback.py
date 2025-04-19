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
        """Initialize the fuzzy feedback system with improved membership functions and rules."""
        # Create fuzzy variables with better defined parameters
        self.setup_fuzzy_system()
        
        # Improved feedback message templates for professionals
        self.feedback_templates = {
            'wrist_stability': [
                "Focus on stabilizing your wrist position.",
                "Maintain consistent wrist alignment through trigger pull.",
                "Reduce wrist movement for improved shot consistency."
            ],
            'elbow_stability': [
                "Maintain stable elbow position to support your aim.",
                "Control support arm motion through trigger pull.",
                "Minimize elbow drift for better accuracy."
            ],
            'stance': [
                "Adjust stance to maintain optimal center of gravity.",
                "Distribute weight evenly for improved stability.",
                "Establish a more stable base position."
            ],
            'head_position': [
                "Maintain consistent cheek weld and head position.",
                "Reduce head movement to stabilize sight picture.",
                "Keep head position fixed relative to sight alignment."
            ],
            'follow_through': [
                "Excellent follow-through maintaining sight alignment.",
                "Maintain position through recoil for optimal performance.",
                "Good shot execution with proper follow-through technique."
            ],
            'general_posture': [
                "Optimize overall body alignment for better stability.",
                "Maintain consistent body position through your shot cycle.",
                "Control breathing and posture for improved accuracy."
            ]
        }

    def setup_fuzzy_system(self):
        """Set up the fuzzy control system with improved variables, membership functions, and rules."""
        # Define fuzzy variables (universes of discourse)
        
        # Sway velocities for different joints (mm/s) - adjusted ranges for better precision
        wrist_sway = ctrl.Antecedent(np.arange(0, 21, 0.5), 'wrist_sway')
        elbow_sway = ctrl.Antecedent(np.arange(0, 21, 0.5), 'elbow_sway')
        nose_sway = ctrl.Antecedent(np.arange(0, 21, 0.5), 'nose_sway')
        
        # Postural stability (px) - adjusted ranges
        hip_dev_x = ctrl.Antecedent(np.arange(0, 31, 0.5), 'hip_dev_x')
        nose_dev_y = ctrl.Antecedent(np.arange(0, 31, 0.5), 'nose_dev_y')
        
        # Follow-through score (0-1) - finer granularity
        follow_through = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'follow_through')
        
        # Output variable: feedback score (0-100) - finer granularity
        feedback_score = ctrl.Consequent(np.arange(0, 101, 1), 'feedback_score')
        
        # Define membership functions for inputs with better calibrated ranges for professional shooters
        
        # Sway velocities - refined thresholds based on professional performance
        wrist_sway['low'] = fuzz.trapmf(wrist_sway.universe, [0, 0, 2, 5])
        wrist_sway['medium'] = fuzz.trimf(wrist_sway.universe, [3, 6, 10])
        wrist_sway['high'] = fuzz.trapmf(wrist_sway.universe, [8, 12, 20, 20])
        
        elbow_sway['low'] = fuzz.trapmf(elbow_sway.universe, [0, 0, 1.5, 4])
        elbow_sway['medium'] = fuzz.trimf(elbow_sway.universe, [2.5, 5, 9])
        elbow_sway['high'] = fuzz.trapmf(elbow_sway.universe, [7, 10, 20, 20])
        
        nose_sway['low'] = fuzz.trapmf(nose_sway.universe, [0, 0, 0.8, 2.5])
        nose_sway['medium'] = fuzz.trimf(nose_sway.universe, [1.5, 3.5, 6])
        nose_sway['high'] = fuzz.trapmf(nose_sway.universe, [5, 7, 20, 20])
        
        # Postural stability - refined thresholds
        hip_dev_x['low'] = fuzz.trapmf(hip_dev_x.universe, [0, 0, 4, 8])
        hip_dev_x['medium'] = fuzz.trimf(hip_dev_x.universe, [6, 10, 16])
        hip_dev_x['high'] = fuzz.trapmf(hip_dev_x.universe, [14, 18, 30, 30])
        
        nose_dev_y['low'] = fuzz.trapmf(nose_dev_y.universe, [0, 0, 2, 6])
        nose_dev_y['medium'] = fuzz.trimf(nose_dev_y.universe, [4, 8, 12])
        nose_dev_y['high'] = fuzz.trapmf(nose_dev_y.universe, [10, 14, 30, 30])
        
        # Follow-through score - more precise thresholds for professionals
        follow_through['poor'] = fuzz.trapmf(follow_through.universe, [0, 0, 0.25, 0.45])
        follow_through['average'] = fuzz.trimf(follow_through.universe, [0.35, 0.55, 0.75])
        follow_through['excellent'] = fuzz.trapmf(follow_through.universe, [0.65, 0.8, 1, 1])
        
        # Define membership functions for output with better calibration
        feedback_score['poor'] = fuzz.trapmf(feedback_score.universe, [0, 0, 25, 40])
        feedback_score['average'] = fuzz.trimf(feedback_score.universe, [30, 50, 70])
        feedback_score['good'] = fuzz.trimf(feedback_score.universe, [60, 75, 90])
        feedback_score['excellent'] = fuzz.trapmf(feedback_score.universe, [80, 90, 100, 100])
        
        # Define improved fuzzy rules with better weighting for professionals
        
        # Rule 1: Wrist and elbow stability (critical for shooting)
        rule1 = ctrl.Rule(
            wrist_sway['high'] | elbow_sway['high'],
            feedback_score['poor']
        )
        
        # Rule 2: Stance stability
        rule2 = ctrl.Rule(
            hip_dev_x['medium'] | hip_dev_x['high'],
            feedback_score['average']
        )
        
        # Rule 3: Head position (critical for sight alignment)
        rule3 = ctrl.Rule(
            nose_dev_y['medium'] | nose_dev_y['high'] | nose_sway['high'],
            feedback_score['average']
        )
        
        # Rule 4: Good follow-through (essential for accuracy)
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
        
        # Rule 7: Excellent stability (professional level)
        rule7 = ctrl.Rule(
            (wrist_sway['low'] & elbow_sway['low'] & nose_sway['low'] & 
            hip_dev_x['low'] & nose_dev_y['low'] & follow_through['excellent']),
            feedback_score['excellent']
        )
        
        # Additional rules for professional-level feedback
        
        # Rule 8: Prioritize wrist stability over elbow
        rule8 = ctrl.Rule(
            (wrist_sway['low'] & elbow_sway['medium'] & follow_through['average']),
            feedback_score['good']
        )
        
        # Rule 9: Head stability is critical
        rule9 = ctrl.Rule(
            (nose_sway['low'] & nose_dev_y['low'] & follow_through['average']),
            feedback_score['good']
        )
        
        # Rule 10: Poor follow-through even with good stability is problematic
        rule10 = ctrl.Rule(
            (wrist_sway['low'] & elbow_sway['low'] & follow_through['poor']),
            feedback_score['average']
        )
        
        # Create control system with all rules
        self.feedback_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10
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
        Generate professional-level text feedback based on metrics and fuzzy rules.
        
        Args:
            metrics: Dictionary containing stability metrics
            
        Returns:
            Text feedback string with professional terminology
        """
        feedback_items = []
        
        # Extract key metrics with proper validation
        wrist_sway = (metrics.get('sway_velocity', {}).get('LEFT_WRIST', 0) +
                    metrics.get('sway_velocity', {}).get('RIGHT_WRIST', 0)) / 2
        
        elbow_sway = (metrics.get('sway_velocity', {}).get('LEFT_ELBOW', 0) +
                    metrics.get('sway_velocity', {}).get('RIGHT_ELBOW', 0)) / 2
        
        nose_sway = metrics.get('sway_velocity', {}).get('NOSE', 0)
        hip_dev_x = metrics.get('dev_x', {}).get('HIPS', 0)
        nose_dev_y = metrics.get('dev_y', {}).get('NOSE', 0)
        follow_through = metrics.get('follow_through_score', 0)
        
        # Priority-based feedback system (professionals care about the most important issues first)
        
        # Check follow-through (high priority for professionals)
        if follow_through < 0.35:  # Poor follow-through
            feedback_items.append(f"Focus on follow-through: maintain position after trigger break.")
        elif follow_through > 0.7 and wrist_sway < 6 and nose_sway < 3:  # Excellent follow-through
            feedback_items.append(np.random.choice(self.feedback_templates['follow_through']))
        
        # Check head position (high priority)
        if nose_dev_y > 5 or nose_sway > 5:  # Significant head movement
            feedback_items.append(np.random.choice(self.feedback_templates['head_position']))
        
        # Check wrist stability (high priority for precision)
        if wrist_sway > 8:  # High wrist sway
            feedback_items.append(np.random.choice(self.feedback_templates['wrist_stability']))
        
        # Check elbow stability (medium priority)
        if elbow_sway > 7:  # High elbow sway
            feedback_items.append(np.random.choice(self.feedback_templates['elbow_stability']))
        
        # Check stance (medium priority)
        if hip_dev_x > 6:  # Medium or high hip deviation
            feedback_items.append(np.random.choice(self.feedback_templates['stance']))
        
        # If performance is excellent across the board, provide positive reinforcement
        if (follow_through > 0.7 and wrist_sway < 4 and elbow_sway < 4 and 
            nose_sway < 3 and hip_dev_x < 5 and nose_dev_y < 4):
            feedback_items = ["Excellent shot execution. Maintain this stability and follow-through."]
        
        # If no specific feedback generated, give general guidance
        if not feedback_items:
            feedback_items.append(np.random.choice(self.feedback_templates['general_posture']))
        
        # Limit to 2 most important feedback items to keep it concise and actionable
        if len(feedback_items) > 2:
            feedback_items = feedback_items[:2]
        
        # Join feedback items with proper spacing
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