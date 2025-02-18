from typing import List, Dict, Optional
from models.api import Model
from vision.vision_analysis import ScreenAnalysis, InterfaceType
from vision.vision_base import UIElement
from task_execution.data_structures import (
    DetailedStep, 
    TaskContext, 
    StepPlan,
    PlanStepHistory,
    StepAttempt
)
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
            4. Learn from previous attempts and avoid repeating failed approaches
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
                
            # Get current step from current plan
            if context.current_plan is None or context.current_step >= len(context.current_plan.steps):
                logger.info("No more steps in current plan")
                return None
                
            current_plan_step = context.current_plan.steps[context.current_step]
            logger.info(f"Processing plan step: {current_plan_step.description}")
            
            # Get step history
            step_histories = context.get_current_step_history(include_mapped=True)
            
            # Build screen context
            screen_context = self._get_screen_context(vision_analysis)
            
            # Generate step
            step_prompt = self._create_step_prompt(
                current_plan_step=current_plan_step,
                screen_context=screen_context,
                vision_analysis=vision_analysis,
                step_histories=step_histories
            )
            
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
        
    def _create_step_prompt(self,
                          current_plan_step,
                          screen_context: str,
                          vision_analysis: ScreenAnalysis,
                          step_histories: List[PlanStepHistory]) -> str:
        """Create the prompt for step generation"""
        history_context = self._format_step_history(step_histories) if step_histories else "No previous attempts"

        return f"""
        Generate a precise UI interaction step to accomplish this goal.

        Current Plan Step: {current_plan_step.description}
        Required Elements: {', '.join(current_plan_step.precondition.key_elements)}
        Expected Outcome: {current_plan_step.expected_outcome.description if current_plan_step.expected_outcome else 'Unknown'}
        
        Interface Type: {vision_analysis.environment.interface_type.name if vision_analysis.environment else 'Unknown'}
        
        Current Screen Elements:
        {screen_context}
        
        Previous Attempts History:
        {history_context}

        Requirements:
        1. Generate EXACTLY ONE specific UI interaction
        2. Use only currently visible elements
        3. Consider previous failed attempts
        4. Choose a different approach if previous attempts failed
        5. Ensure the action moves towards the expected outcome

        Return as JSON:
        {{
            "description": "Detailed description of what needs to be done and why",
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

    def _format_step_history(self, histories: List[PlanStepHistory]) -> str:
        """Format step history for the prompt"""
        if not histories:
            return "No previous attempts"
            
        result = []
        for history in histories:
            failed_attempts = history.all_failed_attempts
            result.append(f"Total attempts: {len(history.attempts)}")
            result.append(f"Failed attempts: {len(failed_attempts)}")
            
            if failed_attempts:
                result.append("\nFailed approaches tried:")
                for attempt in failed_attempts:
                    result.append(f"""
                    - Action: {', '.join(str(step) for step in attempt.detailed_plan.steps)}
                    - Reason for failure: {attempt.evaluation_reasoning}
                    """)
            
            if history.latest_attempt and history.latest_attempt.step_completed:
                result.append("\nLast successful attempt:")
                result.append(f"""
                - Action: {', '.join(str(step) for step in history.latest_attempt.detailed_plan.steps)}
                - Outcome: {history.latest_attempt.evaluation_reasoning}
                """)
                
        return "\n".join(result)
    
    def _create_detailed_step(self, 
                            step_data: dict,
                            vision_analysis: ScreenAnalysis) -> Optional[DetailedStep]:
        """Convert step data into DetailedStep object"""
        try:
            context_data = self._process_step_context(step_data)
            coordinates = self._get_step_coordinates(step_data, vision_analysis)
            
            return DetailedStep(
                description=step_data.get('description') or "",
                action=step_data.get('action') or "",
                target=step_data.get('target'),
                coordinates=tuple(coordinates) if coordinates else None,
                context=context_data,
                expected_result=step_data.get('expected_result')
            )
        except Exception as e:
            logger.error(f"Error creating detailed step: {str(e)}")
            return None
        
    def _generate_step(self,
                      plan_step,
                      context: TaskContext,
                      vision_analysis: ScreenAnalysis,
                      screen_context: str,
                      step_histories: List[PlanStepHistory]) -> Optional[dict]:
        """Generate detailed step data based on plan step and execution history"""
        try:
            # Build history context
            history_context = ""
            if step_histories:
                history_context = "Previous attempts for this step:\n"
                for history in step_histories:
                    history_context += f"""
                    Step: {history.plan_step_description}
                    Total attempts: {len(history.attempts)}
                    Failed attempts: {len(history.all_failed_attempts)}
                    
                    Previous attempts:
                    """
                    for i, attempt in enumerate(history.attempts, 1):
                        history_context += f"""
                        Attempt {i}:
                        - Plan: {attempt.detailed_plan.plan_description}
                        - Actions: {', '.join(str(step) for step in attempt.detailed_plan.steps)}
                        - Success: {attempt.execution_success}
                        - Completed: {attempt.step_completed}
                        - Reasoning: {attempt.evaluation_reasoning}
                        """

            step_prompt = f"""
            Generate a precise UI interaction step to accomplish this goal.

            Current Plan Step: {plan_step.description}
            Required Elements: {', '.join(plan_step.precondition.key_elements)}
            Expected Outcome: {plan_step.expected_outcome.description if plan_step.expected_outcome else 'Unknown'}
            
            Interface Type: {vision_analysis.environment.interface_type.name if vision_analysis.environment else 'Unknown'}
            Current Screen Elements:
            {screen_context}
            
            {history_context}

            Consider:
            1. Previously failed attempts and why they failed
            2. What approaches haven't been tried yet
            3. Whether a different strategy is needed

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
        if vision_analysis.environment and vision_analysis.environment.interface_type in [InterfaceType.TERMINAL, InterfaceType.TEXT_UI]:
            return f"""
            Summary: {vision_analysis.text_screen_info.summary if vision_analysis.text_screen_info else 'No summary available'}
            Screen Description: {vision_analysis.text_screen_info.screen_description if vision_analysis.text_screen_info else 'No screen description available'}
            Available Actions: {self._format_text_interactions(vision_analysis.text_screen_info.visible_interactions) if vision_analysis.text_screen_info else 'No available actions'}
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

    def _get_step_coordinates(self, step_data: dict, vision_analysis: ScreenAnalysis) -> Optional[tuple]:
        """Get coordinates for a step's target element if needed"""
        target = step_data.get('target')
        if target and vision_analysis.capabilities and vision_analysis.capabilities.has_mouse:
            target_elem = self._find_ui_element(target, vision_analysis.elements)
            if target_elem:
                coordinates = vision_analysis.get_coordinates(target_elem)
                if isinstance(coordinates, dict):
                    return tuple(coordinates.values())
                return coordinates
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

    def _generate_complexity_prompt(self, context: TaskContext, screen_context: str, interface_type: InterfaceType) -> str:
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
        
        Goal: {context.user_task}  # Updated from context.goal.description
        Current Phase: {context.current_phase}
        Interface Type: {interface_type.name}
        
        The main UI elements with known coordinates are:
        {screen_context}
        
        Recent Observations:
        {self._format_observations(context.observations)}
        
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
                        context: TaskContext,
                        screen_context: str,
                        interface_type: InterfaceType,
                        require_single_step: bool,
                        plan_description: str,
                        expected_outcome: str) -> str:
        """Generate the prompt for step generation"""
        return f"""
        {f'Generate a SINGLE atomic step.' if require_single_step else 'Break down the required action into individual steps.'}
        
        Goal: {context.user_task}  # Updated from context.goal.description
        Current Phase: {context.current_phase}
        Interface Type: {interface_type.name}
        
        Plan Description: {plan_description}
        Expected Outcome: {expected_outcome}
        
        Current Screen State:
        {screen_context}
        
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
