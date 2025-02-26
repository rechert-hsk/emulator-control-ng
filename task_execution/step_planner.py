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

        mouse_actions = """
        Mouse Click Actions:
        - Simple click: "click" with button="left|right|middle"
        - Double click: "click" with click_type="double" 
        - Click and hold: "click" with hold=true
        - Click and drag: "drag" with from_target and to_target
        """

        return f"""
        You are a UI automation expert that converts high-level tasks into precise, atomic UI interactions.

        YOUR TASK: Generate a single UI interaction step to accomplish this goal:
        {current_plan_step.description}

        CONSTRAINTS:
        - Required elements must be present: {', '.join(current_plan_step.precondition.key_elements)}
        - Expected outcome: {current_plan_step.expected_outcome.description if current_plan_step.expected_outcome else 'Unknown'}
        - Must use only elements that are currently visible
        - Must be exactly ONE specific interaction (click, drag, type, press key, or select)

        CONTEXT:
        Interface: {vision_analysis.environment.interface_type.name if vision_analysis.environment else 'Unknown'}

        {mouse_actions}

        Available Elements:
        {screen_context}

        Previous Attempts:
        {history_context}

        CRITICAL ANALYSIS INSTRUCTIONS:
        1. FIRST carefully analyze the screenshot to understand the current UI state
        2. Identify which UI elements are active, disabled, or in a specific state (selected, hovered, etc.)
        3. Consider the visual context - what phase of the process is shown, what options are available
        4. Determine if required elements from previous failed attempts are now visible or accessible
        5. Check if any new UI elements have appeared that might be relevant

        PLANNING INSTRUCTIONS:
        1. Based on your analysis of the current screenshot state, identify the most appropriate action
        2. If previous attempts failed, choose a different approach based on what's currently visible
        3. Generate a single, specific interaction that moves towards the goal
        4. Format response as JSON with this exact schema:
        {{
            "description": "Clear explanation of what will be done and why based on current UI state",
            "action": "<click|drag|type|press_key|select>",
            "target": "id_of_target_element",
            "context": {{
                "button": "left|right|middle",
                "click_type": "single|double",
                "hold": true/false,
                "from_target": "element_id for drag start",
                "to_target": "element_id for drag end", 
                "input_text": "text to type (for text inputs only)",
                "input_format": "format requirements if any",
                "append_enter": true/false,
                "max_length": number,
                "special_characters": ["list", "of", "special", "chars"],
                "additional_notes": "other relevant details"
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

            if step_data.get('action') == 'click':
                # Set defaults for click action
                context_data.setdefault('button', 'left')
                context_data.setdefault('click_type', 'single') 
                context_data.setdefault('hold', False)

                # Validate button value
                valid_buttons = ['left', 'right', 'middle']
                if context_data['button'] not in valid_buttons:
                    raise ValueError(f"Invalid button value: {context_data['button']}. Must be one of {valid_buttons}")

                # Validate click_type value 
                valid_click_types = ['single', 'double']
                if context_data['click_type'] not in valid_click_types:
                    raise ValueError(f"Invalid click_type: {context_data['click_type']}. Must be one of {valid_click_types}")

            elif step_data.get('action') == 'drag':
                required_fields = ['from_target', 'to_target']
                missing_fields = [f for f in required_fields if f not in context_data]

                if missing_fields:
                    raise ValueError(f"Drag context missing required fields: {missing_fields}")
                
                # Set default button for drag
                context_data.setdefault('button', 'left')
                    
            elif step_data.get('action') == 'type':
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

            return context_data
            
        except Exception as e:
            print(f"Failed to process step context: {str(e)}")
            return {}

    def _get_step_coordinates(self, step_data: dict, vision_analysis: ScreenAnalysis) -> Optional[tuple]:
        """Get coordinates for a step's target element if needed"""
        target = step_data.get('target')
        if target and vision_analysis.capabilities and vision_analysis.capabilities.has_mouse:
            target_elem = self._find_ui_element(target, vision_analysis.elements)
            if target_elem:
                coordinates = vision_analysis.get_coordinates(target_elem, True)
                if isinstance(coordinates, dict):
                    return tuple(coordinates.values())
                return coordinates
            else:
                vision_analysis.get_coordinates_by_description(f"{target}")

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
