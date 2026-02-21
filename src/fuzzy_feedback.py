import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class FuzzyFeedback:
    """
    Implements fuzzy logic feedback system for rifle shooting analysis.
    Provides real-time feedback based on stability metrics using Scikit-Fuzzy.
    """
    
    def __init__(self):
        """Initialize the fuzzy feedback system with improved membership functions and rules."""
        # Initialize antecedents and consequents as None first
        self.wrist_sway: Optional[ctrl.Antecedent] = None
        self.elbow_sway: Optional[ctrl.Antecedent] = None
        self.nose_sway: Optional[ctrl.Antecedent] = None
        self.hip_dev_x: Optional[ctrl.Antecedent] = None
        self.nose_dev_y: Optional[ctrl.Antecedent] = None
        self.follow_through: Optional[ctrl.Antecedent] = None
        self.feedback_score: Optional[ctrl.Consequent] = None
        self.feedback_ctrl: Optional[ctrl.ControlSystem] = None
        self.feedback_simulator: Optional[ctrl.ControlSystemSimulation] = None

        # Setup the system
        self.setup_fuzzy_system()
        
        # Improved feedback message templates for professionals
        self.feedback_templates: Dict[str, List[str]] = {
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

    def setup_fuzzy_system(self) -> None:
        """Set up the fuzzy control system with improved variables, membership functions, and rules."""
        # Define fuzzy variables (universes of discourse)
        
        # Sway velocities for different joints (mm/s or relevant unit)
        self.wrist_sway = ctrl.Antecedent(np.arange(0, 21, 0.5), 'wrist_sway')
        self.elbow_sway = ctrl.Antecedent(np.arange(0, 21, 0.5), 'elbow_sway')
        self.nose_sway = ctrl.Antecedent(np.arange(0, 21, 0.5), 'nose_sway')
        
        # Postural stability (px or relevant unit)
        self.hip_dev_x = ctrl.Antecedent(np.arange(0, 31, 0.5), 'hip_dev_x')
        self.nose_dev_y = ctrl.Antecedent(np.arange(0, 31, 0.5), 'nose_dev_y')
        
        # Follow-through score (0-1)
        self.follow_through = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'follow_through')
        
        # Output variable: feedback score (0-100)
        self.feedback_score = ctrl.Consequent(np.arange(0, 101, 1), 'feedback_score')
        
        # --- Membership Functions ---
        
        # Sway velocities (0-20 scale)
        # Low: Stable hold. High: Unstable.
        # Extended upper bounds to 21 to cover the universe max (20.5)
        self.wrist_sway['low'] = fuzz.trapmf(self.wrist_sway.universe, [0, 0, 2, 5])
        self.wrist_sway['medium'] = fuzz.trimf(self.wrist_sway.universe, [3, 6, 10])
        self.wrist_sway['high'] = fuzz.trapmf(self.wrist_sway.universe, [8, 12, 21, 21])
        
        self.elbow_sway['low'] = fuzz.trapmf(self.elbow_sway.universe, [0, 0, 1.5, 4])
        self.elbow_sway['medium'] = fuzz.trimf(self.elbow_sway.universe, [2.5, 5, 9])
        self.elbow_sway['high'] = fuzz.trapmf(self.elbow_sway.universe, [7, 10, 21, 21])
        
        self.nose_sway['low'] = fuzz.trapmf(self.nose_sway.universe, [0, 0, 0.8, 2.5])
        self.nose_sway['medium'] = fuzz.trimf(self.nose_sway.universe, [1.5, 3.5, 6])
        self.nose_sway['high'] = fuzz.trapmf(self.nose_sway.universe, [5, 7, 21, 21])
        
        # Postural stability (0-30 scale)
        # Extended upper bounds to 31 to cover the universe max (30.5)
        self.hip_dev_x['low'] = fuzz.trapmf(self.hip_dev_x.universe, [0, 0, 4, 8])
        self.hip_dev_x['medium'] = fuzz.trimf(self.hip_dev_x.universe, [6, 10, 16])
        self.hip_dev_x['high'] = fuzz.trapmf(self.hip_dev_x.universe, [14, 18, 31, 31])
        
        self.nose_dev_y['low'] = fuzz.trapmf(self.nose_dev_y.universe, [0, 0, 2, 6])
        self.nose_dev_y['medium'] = fuzz.trimf(self.nose_dev_y.universe, [4, 8, 12])
        self.nose_dev_y['high'] = fuzz.trapmf(self.nose_dev_y.universe, [10, 14, 31, 31])
        
        # Follow-through score (0-1)
        self.follow_through['poor'] = fuzz.trapmf(self.follow_through.universe, [0, 0, 0.25, 0.45])
        self.follow_through['average'] = fuzz.trimf(self.follow_through.universe, [0.35, 0.55, 0.75])
        self.follow_through['excellent'] = fuzz.trapmf(self.follow_through.universe, [0.65, 0.8, 1, 1])
        
        # Feedback Score (0-100)
        self.feedback_score['poor'] = fuzz.trapmf(self.feedback_score.universe, [0, 0, 25, 40])
        self.feedback_score['average'] = fuzz.trimf(self.feedback_score.universe, [30, 50, 70])
        self.feedback_score['good'] = fuzz.trimf(self.feedback_score.universe, [60, 75, 90])
        self.feedback_score['excellent'] = fuzz.trapmf(self.feedback_score.universe, [80, 90, 100, 100])
        
        # --- Fuzzy Rules ---
        
        # Rule 1: High arm sway -> Poor
        rule1 = ctrl.Rule(
            self.wrist_sway['high'] | self.elbow_sway['high'],
            self.feedback_score['poor']
        )
        
        # Rule 2: Unstable stance (hip dev) -> Average (unless arms are worse)
        rule2 = ctrl.Rule(
            self.hip_dev_x['medium'] | self.hip_dev_x['high'],
            self.feedback_score['average']
        )
        
        # Rule 3: Head instability (Nose dev or sway) -> Average
        rule3 = ctrl.Rule(
            self.nose_dev_y['medium'] | self.nose_dev_y['high'] | self.nose_sway['high'],
            self.feedback_score['average']
        )
        
        # Rule 4: Perfect execution -> Excellent
        rule4 = ctrl.Rule(
            self.follow_through['excellent'] & self.nose_sway['low'] & self.wrist_sway['low'] & self.elbow_sway['low'],
            self.feedback_score['excellent']
        )
        
        # Rule 5: Moderate instability -> Average
        rule5 = ctrl.Rule(
            (self.wrist_sway['medium'] & self.elbow_sway['medium']) |
            (self.nose_dev_y['medium'] & self.hip_dev_x['medium']),
            self.feedback_score['average']
        )
        
        # Rule 6: Good stability -> Good
        rule6 = ctrl.Rule(
            (self.wrist_sway['low'] & self.elbow_sway['low'] & self.nose_sway['medium']) |
            (self.follow_through['average'] & self.hip_dev_x['low']),
            self.feedback_score['good']
        )
        
        # Rule 7: Professional stability -> Excellent
        rule7 = ctrl.Rule(
            (self.wrist_sway['low'] & self.elbow_sway['low'] & self.nose_sway['low'] &
            self.hip_dev_x['low'] & self.nose_dev_y['low'] & self.follow_through['excellent']),
            self.feedback_score['excellent']
        )
        
        # Rule 8: Good wrist but medium elbow (Prioritize wrist) -> Good
        rule8 = ctrl.Rule(
            (self.wrist_sway['low'] & self.elbow_sway['medium'] & self.follow_through['average']),
            self.feedback_score['good']
        )
        
        # Rule 9: Stable Head -> Good (Head is critical)
        rule9 = ctrl.Rule(
            (self.nose_sway['low'] & self.nose_dev_y['low'] & self.follow_through['average']),
            self.feedback_score['good']
        )
        
        # Rule 10: Poor follow-through -> Average (even if stable)
        rule10 = ctrl.Rule(
            (self.wrist_sway['low'] & self.elbow_sway['low'] & self.follow_through['poor']),
            self.feedback_score['average']
        )
        
        # Rule 11: Very poor follow-through -> Poor
        rule11 = ctrl.Rule(
            self.follow_through['poor'] & (self.wrist_sway['medium'] | self.elbow_sway['medium']),
            self.feedback_score['poor']
        )

        # Create control system
        self.feedback_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11
        ])
        
        # Create simulator
        self.feedback_simulator = ctrl.ControlSystemSimulation(self.feedback_ctrl)
    
    def generate_feedback(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feedback based on stability metrics.
        
        Args:
            metrics: Dictionary containing stability metrics
            
        Returns:
            Dictionary with feedback score and text
        """
        # Robustly extract metrics with default values
        sway_data = metrics.get('sway_velocity') or {}
        dev_x_data = metrics.get('dev_x') or {}
        dev_y_data = metrics.get('dev_y') or {}

        # Helper to safely get value
        def get_val(data, key, default=0.0):
            val = data.get(key)
            return val if val is not None else default

        try:
            # Sway velocity for joints
            left_wrist = get_val(sway_data, 'LEFT_WRIST')
            right_wrist = get_val(sway_data, 'RIGHT_WRIST')
            wrist_sway = (left_wrist + right_wrist) / 2
            
            left_elbow = get_val(sway_data, 'LEFT_ELBOW')
            right_elbow = get_val(sway_data, 'RIGHT_ELBOW')
            elbow_sway = (left_elbow + right_elbow) / 2
            
            nose_sway = get_val(sway_data, 'NOSE')
            
            # Postural stability
            hip_dev_x = get_val(dev_x_data, 'HIPS')
            nose_dev_y = get_val(dev_y_data, 'NOSE')
            
            # Follow-through score
            follow_through = metrics.get('follow_through_score')
            if follow_through is None:
                follow_through = 0.5
            
            # Input values to fuzzy system (clamped to universe ranges)
            self.feedback_simulator.input['wrist_sway'] = min(wrist_sway, 20)
            self.feedback_simulator.input['elbow_sway'] = min(elbow_sway, 20)
            self.feedback_simulator.input['nose_sway'] = min(nose_sway, 20)
            self.feedback_simulator.input['hip_dev_x'] = min(hip_dev_x, 30)
            self.feedback_simulator.input['nose_dev_y'] = min(nose_dev_y, 30)
            self.feedback_simulator.input['follow_through'] = max(0, min(follow_through, 1))
            
            # Compute result
            self.feedback_simulator.compute()
            
            # Get defuzzified result
            score = self.feedback_simulator.output['feedback_score']
            
            # Generate text feedback using the same input values
            # Pass clamped values to ensure consistency with the simulator
            input_values = {
                'wrist_sway': min(wrist_sway, 20),
                'elbow_sway': min(elbow_sway, 20),
                'nose_sway': min(nose_sway, 20),
                'hip_dev_x': min(hip_dev_x, 30),
                'nose_dev_y': min(nose_dev_y, 30),
                'follow_through': max(0, min(follow_through, 1))
            }
            feedback_text = self._generate_text_feedback(input_values)
            
            return {
                'score': score,
                'text': feedback_text
            }
            
        except Exception as e:
            logger.error(f"Error generating fuzzy feedback: {e}", exc_info=True)
            # Return default feedback if there's an error
            return {
                'score': 50,
                'text': "Keep your body stable and maintain consistent posture.",
                'error': str(e)
            }
    
    def _get_membership(self, antecedent: ctrl.Antecedent, value: float, label: str) -> float:
        """
        Calculate membership degree of a value in a fuzzy set.
        
        Args:
            antecedent: The fuzzy antecedent variable
            value: The input value
            label: The linguistic label ('low', 'medium', 'high', etc.)
            
        Returns:
            Membership degree (0.0 to 1.0)
        """
        if antecedent is None:
            return 0.0
        
        # Clamp value to universe
        value = max(antecedent.universe.min(), min(antecedent.universe.max(), value))
        
        return fuzz.interp_membership(antecedent.universe, antecedent[label].mf, value)

    def _generate_text_feedback(self, inputs: Dict[str, float]) -> str:
        """
        Generate professional-level text feedback based on inputs and membership degrees.
        Decoupled from hardcoded thresholds by using fuzzy membership lookups.
        
        Args:
            inputs: Dictionary containing extracted input values

        Returns:
            Text feedback string with professional terminology
        """
        feedback_items = []
        
        # Check specific issues using membership functions
        # This ensures text feedback aligns perfectly with the score logic
        
        # 1. Follow-Through (High Priority)
        ft_val = inputs['follow_through']
        ft_poor = self._get_membership(self.follow_through, ft_val, 'poor')
        ft_exc = self._get_membership(self.follow_through, ft_val, 'excellent')
        
        if ft_poor > 0.6:
            feedback_items.append("Focus on follow-through: maintain position after trigger break.")
        elif ft_exc > 0.6:
            # Only praise if other things aren't terrible
            pass # Will be handled in "Excellent" check below

        # 2. Head/Nose Stability (High Priority)
        nose_sway_high = self._get_membership(self.nose_sway, inputs['nose_sway'], 'high')
        nose_dev_high = self._get_membership(self.nose_dev_y, inputs['nose_dev_y'], 'high')
        
        if nose_sway_high > 0.5 or nose_dev_high > 0.5:
            feedback_items.append(np.random.choice(self.feedback_templates['head_position']))

        # 3. Wrist Stability (High Priority)
        wrist_high = self._get_membership(self.wrist_sway, inputs['wrist_sway'], 'high')
        if wrist_high > 0.5:
             feedback_items.append(np.random.choice(self.feedback_templates['wrist_stability']))

        # 4. Elbow Stability (Medium Priority)
        elbow_high = self._get_membership(self.elbow_sway, inputs['elbow_sway'], 'high')
        if elbow_high > 0.6 and wrist_high < 0.5: # Don't spam if wrist is already bad
            feedback_items.append(np.random.choice(self.feedback_templates['elbow_stability']))

        # 5. Stance/Hip Stability (Medium Priority)
        hip_high = self._get_membership(self.hip_dev_x, inputs['hip_dev_x'], 'high')
        if hip_high > 0.6:
            feedback_items.append(np.random.choice(self.feedback_templates['stance']))

        # Check for Excellence
        # If no complaints so far and everything looks good
        if not feedback_items:
            # Check if everything is low/excellent
            wrist_low = self._get_membership(self.wrist_sway, inputs['wrist_sway'], 'low')
            elbow_low = self._get_membership(self.elbow_sway, inputs['elbow_sway'], 'low')
            nose_low = self._get_membership(self.nose_sway, inputs['nose_sway'], 'low')

            if (ft_exc > 0.5 and wrist_low > 0.5 and elbow_low > 0.5 and nose_low > 0.5):
                feedback_items.append("Excellent shot execution. Maintain this stability and follow-through.")
            else:
                # General advice if neither bad nor excellent (Average zone)
                feedback_items.append(np.random.choice(self.feedback_templates['general_posture']))
        
        # Limit to 2 most important feedback items
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
