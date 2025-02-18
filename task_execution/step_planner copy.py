from typing import List, Dict, Optional
from models.api import Model
from vision.vision_analysis import ScreenAnalysis, InterfaceType
from vision.vision_base import UIElement
from task_execution.data_structures import DetailedStep, TaskContext, StepPlan
from util.logger import plan_logger as logger
import json

class StepPlanner:
    """Plans and generates detailed execution steps"""
    
    def __init__(self, model):
        self.model = model
        self.system_prompt = """You are an expert UI process automation planner.
            Your role is to:
            1. Break down tasks into clear atomic steps
            2. Generate precise interaction instructions
            3. Consider system constraints and capabilities
            4. Provide clear success criteria for each step
            """

    def determine_next_action(self, 
                        context: 'TaskContext',
                        vision_analysis: ScreenAnalysis) -> Optional[StepPlan]:
        """Determine next action(s) based on current screen analysis and task context"""
        try:
            vision_analysis.analyze_text_screen()
            screen_context = self._get_screen_context(vision_analysis)
            
            # First analyze task complexity
            complexity_data = self._analyze_task_complexity(
                context, 
                vision_analysis, 
                screen_context
            )
            
            # Generate step(s) based on task analysis
            try:
                steps_data = self._generate_steps(
                    context,
                    vision_analysis, 
                    screen_context,
                    not complexity_data.get('requires_multiple_steps', False),
                    complexity_data
                )
                
                detailed_steps = self._create_detailed_steps(steps_data, vision_analysis)
                
                return StepPlan(
                    steps=detailed_steps,
                    plan_description=complexity_data.get('plan_description', ''),
                    expected_outcome=complexity_data.get('expected_outcome', '')
                )
                
            except Exception as e:
                print(f"Failed to generate steps: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Failed to determine next action: {str(e)}")
            return None
    

    def _analyze_task_complexity(self, 
                           context: 'TaskContext',
                           vision_analysis: ScreenAnalysis, 
                           screen_context: str) -> dict:
        """Analyze whether the task requires multiple steps and provide a high-level plan"""
        try:
            complexity_prompt = f"""
            {self._generate_complexity_prompt(
                context,
                screen_context, 
                vision_analysis.environment.interface_type
            )}

            IMPORTANT GUIDELINES:
            1. Stay focused on the IMMEDIATE goal and ignore irrelevant UI elements
            2. Plan should END when screen is expected to change (e.g. new window opens)
            3. Only consider UI elements that directly help accomplish the goal
            4. Each step must have a clear purpose toward the goal
            5. If a step will cause navigation/screen change, it must be the FINAL step
            
            Focus Questions:
            - Does this element directly help achieve the goal?
            - Will this action cause a screen change?
            - Is this the minimal sequence needed?
            """
        
            response = self.model.generate_content(complexity_prompt, vision_analysis.image, self.system_prompt)
            
            if response is None:
                raise ValueError("LLM returned null response for complexity analysis")
                
            complexity_data = Model.parse_json(response)
            
            # Add validation to ensure screen changes are handled properly
            if complexity_data.get('within_current_screen') == False and len(complexity_data.get('steps', [])) > 1:
                # If screen will change, limit to single step
                complexity_data['requires_multiple_steps'] = False
                complexity_data['estimated_steps'] = 1
                complexity_data['explanation'] = "Limited to single step due to expected screen change"

            logger.debug(f"Complexity analysis response: {json.dumps(complexity_data, indent=2)}")
            
            # Validate expected fields
            required_fields = [
                'requires_multiple_steps', 
                'estimated_steps', 
                'explanation',
                'within_current_screen', 
                'limitations',
                'plan_description',  # New field
                'expected_outcome'   # New field
            ]
            missing_fields = [field for field in required_fields if field not in complexity_data]
            
            if missing_fields:
                raise ValueError(f"Complexity analysis response missing required fields: {missing_fields}")
                
            if not isinstance(complexity_data['requires_multiple_steps'], bool):
                raise ValueError("'requires_multiple_steps' must be a boolean")
                
            if not isinstance(complexity_data['estimated_steps'], (int, float)):
                raise ValueError("'estimated_steps' must be a number")
                
            return complexity_data
            
        except Exception as e:
            print(f"Failed to analyze task complexity: {str(e)}")
            return {
                'requires_multiple_steps': False,
                'estimated_steps': 1,
                'explanation': 'Error analyzing complexity',
                'within_current_screen': True,
                'limitations': 'Error analyzing limitations',
                'plan_description': 'Error analyzing plan',
                'expected_outcome': 'Error analyzing outcome'
            }

    def _generate_steps(self,
                    context: 'TaskContext', 
                    vision_analysis: ScreenAnalysis,
                    screen_context: str,
                    require_single_step: bool,
                    complexity_data: dict) -> list:
        """Generate raw steps data using complexity analysis"""
        try:
            steps_prompt = f"""
            {self._generate_steps_prompt(
                context,
                screen_context,
                vision_analysis.environment.interface_type,
                require_single_step, 
                complexity_data.get('plan_description', ''),
                complexity_data.get('expected_outcome', '')
            )}
            
            IMPORTANT GUIDELINES:
            1. Focus ONLY on UI elements that directly help achieve the goal
            2. Ignore any elements that are not relevant to the current goal
            3. If an action will change/navigate screens, it MUST be the final step
            4. Each step must make clear progress toward the goal
            5. Steps must be in optimal order with no unnecessary actions
            
            Ask for each step:
            - Is this action necessary for the goal?
            - Will this cause a screen change?
            - Is there a more direct way to achieve this?
            """
            
            response = self.model.generate_content(steps_prompt, vision_analysis.image, self.system_prompt)

            if response is None:
                raise ValueError("LLM returned null response for step generation")
                
            steps_data = Model.parse_json(response)
            
            # Ensure steps_data is a list
            if isinstance(steps_data, dict):
                steps_data = [steps_data]
            elif not isinstance(steps_data, list):
                raise ValueError(f"Steps data must be a list or dict, got {type(steps_data)}")

            # Validate each step is a dictionary
            for i, step in enumerate(steps_data):
                if not isinstance(step, dict):
                    raise ValueError(f"Step {i} must be a dictionary, got {type(step)}")

            # Validate and filter steps that would occur after screen change
            validated_steps = []
            screen_changing_actions = ['click', 'submit', 'navigate']
            screen_change_detected = False
            
            for step in steps_data:
                if screen_change_detected:
                    break
                    
                action = str(step.get('action', '')).lower()
                if any(change_action in action for change_action in screen_changing_actions):
                    screen_change_detected = True
                
                validated_steps.append(step)
                
            logger.debug(f"Step generation response: {json.dumps(validated_steps, indent=2)}")
            
            # Validate required fields
            required_step_fields = ['description', 'action', 'target', 'context']
            for i, step in enumerate(validated_steps):
                missing_fields = [field for field in required_step_fields if field not in step]
                if missing_fields:
                    raise ValueError(f"Step {i} missing required fields: {missing_fields}")
            
            return validated_steps
            
        except Exception as e:
            print(f"Failed to generate steps: {str(e)}")
            return []

    def _create_detailed_steps(self, 
                             steps_data: list,
                             vision_analysis: ScreenAnalysis) -> List[DetailedStep]:
        """Convert raw steps data into DetailedStep objects"""
        detailed_steps = []
        
        for step_data in steps_data:
            context_data = self._process_step_context(step_data)
            coordinates = self._get_step_coordinates(step_data, vision_analysis)
            
            detailed_steps.append(DetailedStep(
                description=step_data.get('description'),
                action=step_data.get('action'),
                target=step_data.get('target'),
                coordinates=coordinates,
                context=context_data,
                expected_result=step_data.get('expected_result')
            ))
            
        return detailed_steps

    def _get_screen_context(self, vision_analysis: ScreenAnalysis) -> str:
        """Generate context description based on interface type"""
        if vision_analysis.environment.interface_type in [InterfaceType.TERMINAL, InterfaceType.TEXT_UI]:
            return f"""
            Summary: {vision_analysis.text_screen_info.summary}
            Screen Description: {vision_analysis.text_screen_info.screen_description}
            Available Actions: {self._format_text_interactions(vision_analysis.text_screen_info.visible_interactions)}
            """
        return self._create_ui_context(vision_analysis.elements)

    def _process_step_context(self, step_data: dict) -> dict:
        """Process and format the context data for a step"""
        try:
            if not isinstance(step_data, dict):
                raise ValueError(f"Step data must be a dictionary, got {type(step_data)}")
                
            context_data = step_data.get('context')
            if context_data is None:
                raise ValueError("Missing required 'context' field in step data")
                
            if step_data.get('action') == 'type':
                required_fields = ['input_text', 'input_format', 'append_enter', 'max_length', 'special_characters']
                missing_fields = [f for f in required_fields if f not in context_data]
                
                if missing_fields:
                    raise ValueError(f"Typing context missing required fields: {missing_fields}")
                    
                return {
                    'input_text': context_data.get('input_text'),
                    'input_format': context_data.get('input_format'),
                    'append_enter': context_data.get('append_enter', False),
                    'max_length': context_data.get('max_length'),
                    'special_characters': context_data.get('special_characters', []),
                    'additional_notes': context_data.get('additional_notes')
                }
                
            return context_data.get('additional_notes')
            
        except Exception as e:
            print(f"Failed to process step context: {str(e)}")
            return {}

    def _get_step_coordinates(self, step_data: dict, vision_analysis: ScreenAnalysis) -> Optional[dict]:
        """Get coordinates for a step's target element if needed"""
        target = step_data.get('target')
        if target and vision_analysis.capabilities.has_mouse:
            target_elem = self._find_ui_element(target, vision_analysis.elements)
            if target_elem:
                return vision_analysis.get_coordinates(target_elem)
        return None

    def _create_ui_context(self, ui_elements: List[UIElement]) -> str:
        """Create structured description of current UI state"""
        formatted_elements = []
        for element in ui_elements:
            element_info = f"- {element.id}: at {element.context}"
            if element.type:
                element_info += f"  Type: {element.type}"
            if element.text:
                element_info += f"  Text: {element.text}"
            if element.visual_description:
                element_info += f"  Description: {element.visual_description}"
            formatted_elements.append(element_info)
        return "\n".join(formatted_elements)

    def _format_text_interactions(self, interactions) -> str:
        """Format text screen interactions"""
        formatted = []
        for interaction in interactions:
            keys = ", ".join(interaction.keys) if interaction.keys else "No keys"
            formatted.append(
                f"- Keys: {keys}\n"
                f"  Action: {interaction.description}\n"
                f"  Location: {interaction.location}"
            )
        return "\n".join(formatted)

    def _find_ui_element(self, target: str, ui_elements: List[UIElement]) -> Optional[UIElement]:
        """Find matching UI element"""
        for element in ui_elements:
            if element.id and target.lower() in element.id.lower():
                return element
        return None

    def _generate_complexity_prompt(self, context: 'TaskContext', screen_context: str, interface_type: InterfaceType) -> str:
        """Generate the prompt for analyzing task complexity"""
        return f"""
        Analyze this task and determine if it requires multiple steps.

        Definition of a step:
        - A step is a single atomic user interaction with the system
        - Examples of single steps:
        * Clicking one button/link
        * Typing a single value into one field
        * Pressing one key or key combination
        * Selecting one item from a dropdown
        * Checking/unchecking one checkbox
        - A step should NOT include:
        * Multiple clicks or interactions
        * Navigation across different screens
        * Complex sequences that require screen changes
        * Actions that require waiting for system responses
        
        Scope limitations:
        - Focus on the CURRENT task 
        - Only consider actions possible on the CURRENT screen
        - Any step must be possible with the currently visible/available UI elements
        
        Goal: {context.goal.description}
        Current Phase: {context.current_phase}
        Interface Type: {interface_type.name}
        
        The main UI elements with known coordinates are:
        {screen_context}
        
        Recent Observations:
        {self._format_observations(context.observations)}
        
        Previous Actions:
        {self._format_actions(context.attempted_actions)}
        
        Return as JSON:
        {{
            "requires_multiple_steps": boolean,
            "estimated_steps": number,
            "explanation": "Detailed explanation of why multiple steps are needed (if applicable)",
            "within_current_screen": boolean,
            "limitations": "Any limitations or concerns about completing the goal on current screen",
            "plan_description": "Brief high-level description of the planned steps",
            "expected_outcome": "Description of what should be achieved after completing all steps"
        }}
        """


    def _generate_steps_prompt(self, 
                         context: 'TaskContext',
                         screen_context: str,
                         interface_type: InterfaceType,
                         require_single_step: bool,
                         plan_description: str,      # Added parameter
                         expected_outcome: str) -> str:    # Added parameter
        """Generate the prompt for step generation"""
        return f"""
        {f'Generate a SINGLE atomic step.' if require_single_step else 'Break down the required action into individual steps.'}
        
        Goal: {context.goal.description}
        Current Phase: {context.current_phase}
        Interface Type: {interface_type.name}
        
        Plan Description: {plan_description}
        Expected Outcome: {expected_outcome}
        
        Current Screen State:
        {screen_context}
        
        Recent Observations:
        {self._format_observations(context.observations)}
        
        Previous Actions:
        {self._format_actions(context.attempted_actions)}
        
        Definition of a SINGLE atomic step:
        - ONE mouse click on a specific element
        - ONE key press or key combination
        - Typing into ONE input field
        - Selecting ONE item from a list/dropdown
        - Checking/unchecking ONE checkbox
        
        For text input actions, you MUST specify:
        - The exact text to be typed
        - Any special characters or formatting required
        - Whether the input should end with Enter/Return
        - Any maximum length restrictions visible on screen
        
        Return as JSON{' array' if not require_single_step else ''} with{' each step containing' if not require_single_step else ''}:
        {{
            "description": "what needs to be done and why",
            "action": "specific UI action (click, type, press_key etc)",
            "target": "element to interact with",
            "context": {{
                "input_text": "exact text to type (for text inputs)",
                "input_format": "any special format requirements", 
                "append_enter": boolean,
                "max_length": number or null,
                "special_characters": ["list", "of", "special", "chars", "needed"],
                "additional_notes": "any other relevant information"
            }}
        }}
        """

    
    def _format_observations(self, observations: List[str]) -> str:
        """Format recent observations for prompt"""
        if not observations:
            return "No recent observations"
        return "\n".join(f"- {obs}" for obs in observations)

    def _format_actions(self, actions: List[str]) -> str:
        """Format action history for prompt"""
        if not actions:
            return "No previous actions"
        return "\n".join(f"- {action}" for action in actions)
