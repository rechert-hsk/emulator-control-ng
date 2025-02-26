import time
from typing import List, Dict, Optional, Any
from models.api import Model
from vision.vision_analysis import ScreenAnalysis
from dataclasses import dataclass
from task_execution.data_structures import PlanCondition, PlanStep, ExecutionPlan, PlanStepHistory, StepAttempt, TaskContext, StepPlan
import traceback
import json

class TaskPlanner:
    """Creates high-level execution plans based on task goals and UI state"""

    def __init__(self, planner_model):
        self.planner_model = planner_model
        
        self.system_prompt = """You are an expert system automation planner.
            Create plans that describe tasks in natural language steps.
            For each step, describe:
            1. What needs to be true before the step (precondition)
            2. What action needs to be taken
            3. What should be true afterwards (expected outcome)
            
            Important rules:
            - Focus on observable UI elements and clear descriptions
            - When you cannot reliably predict an outcome, mark it as unknown
            - A plan MUST end in one of two ways:
              1. The final step achieves the task goal with a clear expected outcome
              2. A step leads to an unknown state, requiring replanning
            - Never leave a plan in an intermediate state"""

    def create_execution_plan(self, task_description: str, screen: ScreenAnalysis, current_context: Optional[TaskContext] = None) -> ExecutionPlan:
        """Create high-level execution plan based on task and current screen"""
        execution_context = ""
        if current_context:
            step_histories = current_context.get_current_step_history(include_mapped=True)
            execution_context = self._build_execution_context(step_histories)

        prompt = f"""
        Task Goal:
        {task_description}

        Current Screen Analysis:
        - Visible Elements: {screen.elements}
        - Screen Type: {screen.environment.interface_type.name if screen.environment else 'Unknown'}
        - Text Content: {screen.ocr_text if hasattr(screen, 'ocr_text') else 'None'}

        Previous Execution History:
        {execution_context}

        Create a precise execution plan that:
        1. Considers historical progress:
        - Don't repeat steps marked as completed unless state changed
        - Learn from failed attempts and try alternative approaches
        - Use previously successful strategies when applicable
        - Skip steps that became irrelevant

        2. Maintains execution continuity:
        - Preserve working steps from previous plans when possible
        - Keep correct dependency ordering (completed -> incomplete)
        - Only generate new steps when necessary
        - Document why steps were kept or changed

        3. Verifies current state:
        - Confirm if state matches expectations
        - Plan recovery if state diverged
        - Document state assumptions
        - Include error handling steps

        Format response as JSON:
        {{
            "initial_state": {{
                "description": "Detailed state including historical context",
                "key_elements": ["list", "of", "currently", "visible", "elements"]
            }},
            "steps": [
                {{
                    "description": "Specific action to take",
                    "precondition": {{
                        "description": "What must be true before step",
                        "key_elements": ["required", "elements"]
                    }},
                    "expected_outcome": {{
                        "description": "What should be true after",
                        "key_elements": ["expected", "elements"]
                    }} or null if unknown
                }}
            ],
            "is_complete": boolean,
            "planning_notes": "Analysis of plan changes and continuity",
            "context_analysis": {{
                "state_matches_expected": boolean,
                "divergence_details": "String if state differs",
                "reused_steps": ["list of steps reused from previous plan"],
                "modified_steps": ["list of steps that were modified"],
                "new_steps": ["list of completely new steps"]
            }}
        }}

        Important:
        - Only include UI elements that are currently visible
        - Document any state divergence that requires recovery
        - Explain why steps were kept, modified, or added
        - Set is_complete=false if more planning needed
        """

        try:
            response = self.planner_model.generate_content(prompt, screen.image, system_prompt=self.system_prompt)
            plan_data = Model.parse_json(response)
            
            # Extract only the fields needed for PlanCondition
            initial_state_data = {
                'description': plan_data['initial_state']['description'],
                'key_elements': plan_data['initial_state']['key_elements']
            }
            
            # Create plan steps from response
            steps = []
            for step in plan_data['steps']:
                # Extract only the fields needed for PlanCondition
                precondition_data = {
                    'description': step['precondition']['description'],
                    'key_elements': step['precondition']['key_elements']
                }
                
                expected_outcome_data = None
                if step.get('expected_outcome'):
                    expected_outcome_data = {
                        'description': step['expected_outcome']['description'],
                        'key_elements': step['expected_outcome']['key_elements']
                    }
                
                steps.append(PlanStep(
                    description=step['description'],
                    precondition=PlanCondition(**precondition_data),
                    expected_outcome=PlanCondition(**expected_outcome_data) if expected_outcome_data else None
                ))

            # Validate plan termination
            if steps:
                last_step = steps[-1]
                if not last_step.expected_outcome and not plan_data.get('is_complete', False):
                    steps.append(PlanStep(
                        description="Analyze new system state and replan next steps",
                        precondition=PlanCondition(
                            description="System is in unknown state after previous action",
                            key_elements=[]
                        ),
                        expected_outcome=None
                    ))
                
            return ExecutionPlan(
                steps=steps,
                initial_state=PlanCondition(**initial_state_data),
                is_complete=plan_data.get('is_complete', False)
            )

        except Exception as e:
            print(f"Failed to create execution plan: {str(e)}")
            raise Exception("Failed to create execution plan")        

    def evaluate_progress(self, 
                 current_context: TaskContext, 
                 screen: ScreenAnalysis, 
                 previous_step_plan: Optional[StepPlan] = None,
                 previous_step_plan_success: bool = False) -> bool:
        """
        Evaluate the current progress and determine next steps.
        Returns True if task is completed successfully, False otherwise.
        """
        import json
        import time
        import traceback
        import re
        from typing import Optional
        
        if current_context.current_plan is None:
            raise ValueError("Current plan is None")
        current_step = current_context.current_plan.steps[current_context.current_step]
        step_histories = current_context.get_current_step_history(include_mapped=True)

        screen_state = {
            'elements': [str(elem) for elem in screen.elements],
            'text_content': screen.ocr_text if hasattr(screen, 'ocr_text') else 'None',
            'interface_type': screen.environment.interface_type.name if screen.environment else 'Unknown',
            'operating system': screen.environment.os if screen.environment else 'Unknown',
            'timestamp': time.time()
        }
        
        execution_context = self._build_execution_context(step_histories)

        prompt = f"""
        You will evaluate whether a step in a multi-step task has been completed based on the current screen state.
        Follow this structured workflow, answering each section in sequence:

        ---TASK INFORMATION---
        Original Goal: {current_context.user_task}
        Current Step: #{current_context.current_step + 1} of {len(current_context.current_plan.steps)}
        Step Description: {current_step.description}
        
        Expected Outcome:
        - Description: {current_step.expected_outcome.description if current_step.expected_outcome else 'Not specified'}
        - Required Elements: {', '.join(current_step.expected_outcome.key_elements) if current_step.expected_outcome and current_step.expected_outcome.key_elements else 'Not specified'}

        Current Screen State:
        - Visible Elements: {screen.elements}
        - Interface Type: {screen.environment.interface_type.name if screen.environment else 'Unknown'}

        Previous Execution History:
        {execution_context}

        ---WORKFLOW STEPS---

        STEP 1: STATE OBSERVATION
        List the key elements currently visible on the screen:
        
        STEP 2: EXPECTED OUTCOME ANALYSIS
        Break down the expected outcome into specific verifiable criteria:
        
        STEP 3: COMPARISON
        For each expected criterion, indicate if it is met by the current state and provide evidence:
        
        STEP 4: GOAL STATE ASSESSMENT
        Determine if the overall task goal has been achieved regardless of the specific step:
        
        STEP 5: DECISION
        Based on your analysis:
        - Is this step completed? (Yes/No)
        - What should be the next action? (Proceed/Replan)
        - Why? (Provide specific reasoning)

        ---FINAL EVALUATION---
        Provide your final evaluation in a valid JSON format exactly as shown below:
        ```json
        {{
            "step_completed": true or false,
            "reasoning": "Detailed explanation of your decision",
            "success_criteria_met": ["List specific criteria that were met"],
            "missing_criteria": ["List specific criteria that were not met"],
            "next_action": "proceed" or "replan",
            "task_goal_achieved": true or false
        }}
        ```

        IMPORTANT NOTES:
        - Focus on whether the OUTCOME has been achieved, not the specific actions taken
        - If the expected state is already present, the step should be marked as complete
        - Some steps may be completed through alternative paths - focus on the result, not the method
        - If the ultimate task goal is achieved, mark task_goal_achieved=true regardless of step completion
        - Your response MUST include the properly formatted JSON block
        """

        try:
            response = self.planner_model.generate_content(prompt, screen.image, system_prompt=self.system_prompt)
            response_text = response
            
            # Extract JSON using regex pattern matching for code blocks
            json_pattern = r"```(?:json)?\s*({.*?})\s*```"
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if json_matches:
                json_str = json_matches[0]
                try:
                    evaluation = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try cleaning up the JSON string if it contains literal "true" or "false" without quotes
                    cleaned_json = re.sub(r':\s*(true|false)\s*([,}])', r': \1\2', json_str)
                    cleaned_json = cleaned_json.replace('true', 'true').replace('false', 'false')
                    try:
                        evaluation = json.loads(cleaned_json)
                    except:
                        # Last resort: extract a simpler evaluation
                        print("Could not parse JSON response. Using fallback evaluation.")
                        # Simple fallback extraction
                        is_completed = "yes" in response_text.lower() and "step completed" in response_text.lower()
                        evaluation = {
                            "step_completed": is_completed,
                            "reasoning": "Extracted from text response due to JSON parsing failure",
                            "success_criteria_met": [],
                            "missing_criteria": [],
                            "next_action": "replan" if not is_completed else "proceed",
                            "task_goal_achieved": False
                        }
            else:
                # No JSON found, extract basic information
                print("No JSON block found in response. Using fallback evaluation.")
                is_completed = "yes" in response_text.lower() and "step completed" in response_text.lower()
                evaluation = {
                    "step_completed": is_completed,
                    "reasoning": "Extracted from text response - no JSON block found",
                    "success_criteria_met": [],
                    "missing_criteria": [],
                    "next_action": "replan" if not is_completed else "proceed",
                    "task_goal_achieved": False
                }

            forced_replan = False
            # Record attempt if we have a previous step plan
            if previous_step_plan:
                attempt = StepAttempt(
                    timestamp=time.time(),
                    detailed_plan=previous_step_plan,
                    execution_success=previous_step_plan_success,
                    step_completed=evaluation["step_completed"],
                    evaluation_reasoning=evaluation["reasoning"],
                    screen_state=screen_state,
                    failure_details=evaluation.get("reasoning") if not evaluation["step_completed"] else None,
                    interaction_result=f"Success criteria met: {', '.join(evaluation.get('success_criteria_met', []))}"
                )
                
                current_context.record_attempt(
                    step_index=current_context.current_step,
                    attempt=attempt  
                )

                print(f"Evaluation result: {evaluation}")

            # First check if overall task is complete
            if evaluation.get("task_goal_achieved", False):
                if current_context.current_plan.is_complete:  
                    return True

            # Handle step progress
            if evaluation["step_completed"]:
                if evaluation["next_action"] == "proceed":
                    print(f"Step {current_context.current_step + 1} completed successfully")
                    if current_context.current_step < len(current_context.current_plan.steps) - 1:  
                        current_context.current_step += 1
                    else:
                        print("Plan ended but we have not reached the task goal")
                        print("Replanning is required")
                        forced_replan = True
                    
            if evaluation["next_action"] == "replan" or forced_replan:
                print("Replanning required")
                print(f"Reason: {evaluation.get('reasoning', 'No analysis provided')}")
                print("Creating new plan...")
                new_plan = self.create_execution_plan(current_context.user_task, screen, current_context)
                new_plan.print_plan()
                
                # Try to map steps from old plan to new plan
                step_mapping = self._map_plans(current_context.current_plan, new_plan)  
                
                current_context.add_new_plan(
                    new_plan=new_plan,
                    reason=evaluation["reasoning"],
                    step_mapping=step_mapping
                )
                current_context.current_step = 0
                return False

            return False

        except Exception as e:
            print(f"Failed to evaluate progress: {str(e)}")
            traceback.print_exc()
            # Provide a fallback response to avoid crashing the system
            return False

    def _build_execution_context(self, step_histories: List[PlanStepHistory]) -> str:
        """Build detailed execution context from step histories"""
        if not step_histories:
            return "No previous attempts"
            
        context = []
        for history in step_histories:
            context.append(f"\nStep '{history.plan_step_description}':")
            context.append(f"Total attempts: {len(history.attempts)}")
            
            for i, attempt in enumerate(history.attempts, 1):
                context.append(f"\nAttempt {i}:")
                context.append(f"- Actions taken:")
                for step in attempt.detailed_plan.steps:
                    context.append(f"  * {str(step)}")
                context.append(f"- Technical success: {attempt.execution_success}")
                context.append(f"- Step completed: {attempt.step_completed}")
                context.append(f"- Reasoning: {attempt.evaluation_reasoning}")
                if attempt.failure_details:
                    context.append(f"- Failure details: {attempt.failure_details}")
        
        return "\n".join(context)

    def _map_plans(self, old_plan: ExecutionPlan, new_plan: ExecutionPlan) -> Dict[int, int]:
        """
        Try to map steps between old and new plans based on similarity.
        Returns a dictionary mapping old step indices to new step indices.
        """
        # This is a simplified version - you might want to use more sophisticated
        # similarity matching based on your specific needs
        mapping = {}
        for old_idx, old_step in enumerate(old_plan.steps):
            for new_idx, new_step in enumerate(new_plan.steps):
                if old_step.description.lower() == new_step.description.lower():
                    mapping[old_idx] = new_idx
                    break
        return mapping
    

    def _build_historical_context(self, context: TaskContext) -> str:
        """Build detailed historical context from previous plans and attempts"""
        if not context.plan_versions:
            return "No historical context available"

        history = []
        history.append("=== Historical Execution Context ===\n")

        # Track completed and failed steps across plans
        completed_steps = set()
        failed_steps = {}  # step description -> list of failure reasons
        successful_strategies = {}  # step type -> successful approach

        for version_idx, plan_version in enumerate(context.plan_versions):
            history.append(f"\nPlan Version {version_idx + 1}")
            
            for step_idx, step in enumerate(plan_version.plan.steps):
                step_key = f"{version_idx}:{step_idx}"
                
                if step_key in context.step_histories:
                    step_history = context.step_histories[step_key]
                    
                    # Analyze attempts
                    success_count = 0
                    failure_reasons = []
                    
                    for attempt in step_history.attempts:
                        if attempt.step_completed:
                            success_count += 1
                            if attempt.detailed_plan:
                                # Record successful strategy
                                step_type = self._categorize_step(step.description)
                                successful_strategies[step_type] = attempt.detailed_plan
                        else:
                            if attempt.failure_details:
                                failure_reasons.append(attempt.failure_details)

                    if success_count > 0:
                        completed_steps.add(step.description)
                        history.append(f"✓ Step {step_idx + 1}: {step.description}")
                        history.append(f"  Completed after {len(step_history.attempts)} attempts")
                    else:
                        failed_steps[step.description] = failure_reasons
                        history.append(f"✗ Step {step_idx + 1}: {step.description}")
                        history.append(f"  Failed {len(step_history.attempts)} times")
                        if failure_reasons:
                            history.append(f"  Failure reasons: {'; '.join(failure_reasons)}")

        # Add analysis summary
        history.append("\n=== Historical Analysis ===")
        history.append(f"\nCompleted Steps: {len(completed_steps)}")
        for step in completed_steps:
            history.append(f"- {step}")
        
        history.append(f"\nFailed Steps: {len(failed_steps)}")
        for step, reasons in failed_steps.items():
            history.append(f"- {step}")
            if reasons:
                history.append(f"  Reasons: {'; '.join(reasons)}")

        if successful_strategies:
            history.append("\nSuccessful Strategies:")
            for step_type, plan in successful_strategies.items():
                history.append(f"- {step_type}: {plan.plan_description}")

        return "\n".join(history)

    def _categorize_step(self, step_description: str) -> str:
        """Categorize step type for strategy matching"""
        # Simple categorization based on keywords
        step_lower = step_description.lower()
        if "click" in step_lower:
            return "button_interaction"
        elif "type" in step_lower or "enter" in step_lower:
            return "text_input"
        elif "select" in step_lower:
            return "selection"
        elif "wait" in step_lower:
            return "wait"
        else:
            return "other"