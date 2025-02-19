import time
from typing import List, Dict, Optional, Any
from models.api import Model
from vision.vision_analysis import ScreenAnalysis
from dataclasses import dataclass
from task_execution.data_structures import PlanCondition, PlanStep, ExecutionPlan, PlanStepHistory, StepAttempt, TaskContext, StepPlan

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

    def create_execution_plan(self, task_description: str, screen: ScreenAnalysis) -> ExecutionPlan:
        """Create high-level execution plan based on task and current screen"""
        prompt = f"""
        Task Goal:
        {task_description}

        Current Screen Analysis:
        - Visible Elements: {screen.elements}
        - Screen Type: {screen.environment.interface_type.name if screen.environment else 'Unknown'}
        - Text Content: {screen.text_analysis if hasattr(screen, 'text_analysis') else 'None'}

        Create a precise execution plan that:
        1. Starts with ONLY the currently visible elements and state
        2. Plans each step based on confirmed visible elements
        3. Avoids assumptions about future states unless explicitly predictable
        4. Requires replanning when entering unknown states

        Rules for Step Planning:
        - Each step must be based on currently visible/confirmed UI elements
        - Only include actions that can be performed with visible elements
        - When an action leads to an unknown state, stop and require replanning
        - Do not assume the presence of elements that aren't confirmed visible
        - For navigation, plan only one screen transition at a time

        Format response as JSON:
        {{
            "initial_state": {{
                "description": "Detailed description of current visible elements and state",
                "key_elements": ["list", "of", "currently", "visible", "elements"]
            }},
            "steps": [
                {{
                    "description": "Specific action to take with visible elements",
                    "precondition": {{
                        "description": "What must be visible/true before this step",
                        "key_elements": ["required", "visible", "elements"]
                    }},
                    "expected_outcome": {{
                        "description": "What should be immediately observable after",
                        "key_elements": ["expected", "visible", "elements"]
                    }} or null if outcome unknown
                }}
            ],
            "is_complete": false if unknown outcome, true if goal achieved
        }}

        Important:
        - Only include UI elements that are currently visible in initial_state
        - Plan steps only based on confirmed visible elements
        - Set expected_outcome to null when next state is unpredictable
        - Set is_complete=false when plan needs replanning
        - Each step must be immediately actionable based on visible elements
        """


        try:
            response = self.planner_model.generate_content(prompt, screen.image, system_prompt=self.system_prompt)
            plan_data = Model.parse_json(response)
            
            steps = []
            for step in plan_data['steps']:
                steps.append(PlanStep(
                    description=step['description'],
                    precondition=PlanCondition(**step['precondition']),
                    expected_outcome=PlanCondition(**step['expected_outcome']) if step.get('expected_outcome') else None
                ))

            # Validate plan termination
            if steps:
                last_step = steps[-1]
                if not last_step.expected_outcome and not plan_data.get('is_complete', False):
                    # Add explicit replan step when ending in unknown state
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
                initial_state=PlanCondition(**plan_data['initial_state']),
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
        if current_context.current_plan is None:
            raise ValueError("Current plan is None")
        current_step = current_context.current_plan.steps[current_context.current_step]
        step_histories = current_context.get_current_step_history(include_mapped=True)

        screen_state = {
            'elements': [str(elem) for elem in screen.elements],
            'text_content': screen.text_analysis if hasattr(screen, 'text_content') else None,
            'interface_type': screen.environment.interface_type.name if screen.environment else 'Unknown',
            'timestamp': time.time()
        }
        

        execution_context = self._build_execution_context(step_histories)

        prompt = f"""
        You are a strict automated testing system evaluating task execution progress. Your evaluation must be precise and based on clear evidence.

        TASK INFORMATION:
        - Original Goal: {current_context.user_task}
        - Current Step: {current_context.current_step + 1} of {len(current_context.current_plan.steps)}
        - Step Description: {current_step.description}
        
        EXPECTED OUTCOME:
        - Description: {current_step.expected_outcome.description if current_step.expected_outcome else 'Unknown'}
        - Required Elements: {', '.join(current_step.expected_outcome.key_elements) if current_step.expected_outcome else 'Unknown'}

        CURRENT STATE:
        Visible Elements: {screen.elements}

        {execution_context}

        EVALUATION RULES:
        1. Step Completion:
           - Mark step_completed=true ONLY if ALL expected elements are present
           - The mere ability to perform an action is NOT completion
           - Visual confirmation of the expected outcome is required

        2. Screen State:
           - Verify all required elements are visible and in the expected state
           - Compare current elements against expected_outcome elements

        3. Task Goal:
           - Only mark task_goal_achieved=true if final objective is verifiably complete
           - Partial progress is not sufficient

        REQUIRED RESPONSE FORMAT:
        {{
            "step_completed": boolean,
            "reasoning": "Detailed comparison of expected vs actual state",
            "next_action": "proceed" or "replan",
            "success_criteria_met": ["list each matched expected element"],
            "missing_criteria": ["list expected elements not found"],
            "task_goal_achieved": boolean,
            "failure_analysis": "Required if step_completed=false: what's missing and why"
        }}

        CRITICAL: Your evaluation must be conservative - when in doubt, mark as incomplete.
        """

        try:
            response = self.planner_model.generate_content(prompt, screen.image, system_prompt=self.system_prompt)
            evaluation = Model.parse_json(response)

            # Record attempt if we have a previous step plan
            if previous_step_plan:
                attempt = StepAttempt(
                    timestamp=time.time(),
                    detailed_plan=previous_step_plan,
                    execution_success=previous_step_plan_success,
                    step_completed=evaluation["step_completed"],
                    evaluation_reasoning=evaluation["reasoning"],
                    screen_state=screen_state,
                    failure_details=evaluation.get("failure_analysis") if not evaluation["step_completed"] else None,
                    interaction_result=f"Success criteria met: {', '.join(evaluation.get('success_criteria_met', []))}"
                )
                
                current_context.record_attempt(
                    step_index=current_context.current_step,
                    attempt=attempt  # Pass the complete StepAttempt object
                )


            # First check if overall task is complete
            if evaluation.get("task_goal_achieved", False):
                if current_context.current_plan.is_complete:  # Fixed this line
                    return True

            # Handle step progress
            if evaluation["step_completed"]:
                if evaluation["next_action"] == "proceed":
                    if current_context.current_step < len(current_context.current_plan.steps) - 1:  # Fixed this line
                        print(f"Step {current_context.current_step + 1} completed successfully")
                        current_context.current_step += 1
                    return False
                    
            if evaluation["next_action"] == "replan":
                print("Replanning required")
                print(f"Reason: {evaluation.get('failure_analysis', 'No analysis provided')}")
                print("new plan")
                new_plan = self.create_execution_plan(current_context.user_task, screen)
                new_plan.print_plan()
                
                # Try to map steps from old plan to new plan
                step_mapping = self._map_plans(current_context.current_plan, new_plan)  # Fixed this line
                
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