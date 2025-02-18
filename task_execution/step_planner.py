from typing import List, Dict, Optional
from models.api import Model
from vision.vision_analysis import ScreenAnalysis, InterfaceType
from vision.vision_base import UIElement
from task_execution.data_structures import DetailedStep, TaskContext, StepPlan
from util.logger import plan_logger as logger
import json

class StepPlanner:
    """Plans and generates detailed execution steps based on the execution plan in TaskContext"""
    
    def __init__(self, model):
        self.model = model
        self.system_prompt = """You are an expert UI process automation planner.
            Your role is to:
            1. Generate precise interaction instructions for the current step
            2. Consider system constraints and capabilities
            3. Provide clear success criteria for the step
            """

    def determine_next_action(self, 
                            context: TaskContext,
                            vision_analysis: ScreenAnalysis) -> Optional[StepPlan]:
        """Determine next action based on current plan step and screen analysis"""
        try:
            # Validate inputs
            if not context or not vision_analysis:
                logger.error("Missing required context or vision_analysis")
                return None
                
            # Get current step from plan
            if context.current_step >= len(context.plan.steps):
                logger.info("No more steps in plan")
                return None
                
            current_plan_step = context.plan.steps[context.current_step]
            logger.info(f"Processing plan step: {current_plan_step.description}")
            
            # Analyze screen
            vision_analysis.analyze_text_screen()
            screen_context = self._get_screen_context(vision_analysis)
            
            # Generate step data
            step_data = self._generate_step(
                current_plan_step,
                context,
                vision_analysis,
                screen_context
            )
            
            if not step_data:
                logger.error("Failed to generate step data")
                return None
                
            # Create detailed step
            detailed_step = self._create_detailed_step(step_data, vision_analysis)
            if not detailed_step:
                logger.error("Failed to create detailed step")
                return None
            
            # Create and return plan
            return StepPlan(
                steps=[detailed_step],
                plan_description=current_plan_step.description,
                expected_outcome=current_plan_step.expected_outcome.description if current_plan_step.expected_outcome else ""
            )
                
        except Exception as e:
            logger.error(f"Error in determine_next_action: {str(e)}")
            return None

    def _generate_step(self,
                      plan_step,
                      context: TaskContext,
                      vision_analysis: ScreenAnalysis,
                      screen_context: str) -> Optional[dict]:
        """Generate detailed step data based on plan step"""
        try:
            step_prompt = f"""
            Generate a precise UI interaction step to accomplish this goal.

            Current Plan Step: {plan_step.description}
            Required Elements: {', '.join(plan_step.precondition.key_elements)}
            Expected Outcome: {plan_step.expected_outcome.description if plan_step.expected_outcome else 'Unknown'}
            
            Interface Type: {vision_analysis.environment.interface_type.name}
            Current Screen Elements:
            {screen_context}
            
            Recent Observations:
            {self._format_observations(context.observations)}

            Previous Actions:
            {self._format_actions(context.attempted_actions)}

            You must return EXACTLY ONE specific UI interaction as JSON:
            {{
                "description": "what needs to be done and why",
                "action": "exactly one of: click, type, press_key, select",
                "target": "specific element to interact with based on visible elements",
                "context": {{
                    "input_text": "exact text to type (for text inputs)",
                    "input_format": "any special format requirements",
                    "append_enter": boolean,
                    "max_length": number,
                    "special_characters": [],
                    "additional_notes": "any other relevant information"
                }}
            }}
            """

            logger.debug(f"Generating step with prompt: {step_prompt}")
            response = self.model.generate_content(step_prompt, vision_analysis.image, self.system_prompt)
            
            if not response:
                logger.error("Model returned empty response")
                return None
                
            step_data = Model.parse_json(response)
            if not step_data:
                logger.error("Failed to parse model response as JSON")
                return None
                
            self._validate_step_data(step_data)
            return step_data
            
        except Exception as e:
            logger.error(f"Error in _generate_step: {str(e)}")
            return None

    def _create_detailed_step(self, 
                            step_data: dict,
                            vision_analysis: ScreenAnalysis) -> Optional[DetailedStep]:
        """Convert step data into DetailedStep object"""
        try:
            context_data = self._process_step_context(step_data)
            coordinates = self._get_step_coordinates(step_data, vision_analysis)
            
            return DetailedStep(
                description=step_data.get('description'),
                action=step_data.get('action'),
                target=step_data.get('target'),
                coordinates=coordinates,
                context=context_data,
                expected_result=step_data.get('expected_result')
            )
        except Exception as e:
            logger.error(f"Error creating detailed step: {str(e)}")
            return None

    def _validate_step_data(self, step_data: dict) -> bool:
        """Validate step data has required fields"""
        required_fields = ['description', 'action', 'target', 'context']
        
        if not isinstance(step_data, dict):
            raise ValueError(f"Step data must be a dictionary, got {type(step_data)}")
            
        missing_fields = [field for field in required_fields if field not in step_data]
        if missing_fields:
            raise ValueError(f"Step missing required fields: {missing_fields}")
            
        valid_actions = ['click', 'type', 'press_key', 'select']
        if step_data['action'] not in valid_actions:
            raise ValueError(f"Invalid action: {step_data['action']}")
            
        return True

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
