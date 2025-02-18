from typing import List, Dict, Optional, Any
from models.api import Model
from vision.vision_analysis import ScreenAnalysis
from dataclasses import dataclass
from task_execution.data_structures import PlanCondition, PlanStep, ExecutionPlan, TaskContext, StepPlan

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
        Create a plan for this task:
        {task_description}

        Current screen shows:
        {screen.elements}

        Create a sequence of steps, where each step includes:
        1. Description of what needs to be done
        2. What should be true/visible before the step
        3. What should be true/visible after (if predictable)

        IMPORTANT:
        The plan must end in one of two ways:
        1. Final step completes the task with clear success criteria
        2. A step leads to an unknown outcome, requiring replanning
        Never leave a plan in an incomplete state.

        Format response as JSON:
        {{
            "initial_state": {{
                "description": "Natural language description of current state",
                "key_elements": ["list", "of", "important", "UI elements"]
            }},
            "steps": [
                {{
                    "description": "Natural language description of what to do",
                    "precondition": {{
                        "description": "What should be true before this step",
                        "key_elements": ["required", "UI", "elements"]
                    }},
                    "expected_outcome": {{
                        "description": "What should be true after",
                        "key_elements": ["expected", "UI", "elements"]
                    }} or null
                }}
            ],
            "is_complete": true/false
        }}

        Set is_complete:
        - true if the final step achieves the task goal
        - false if plan ends with unknown state requiring replanning
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
        
        # Build evaluation context from all relevant history
        execution_context = ""
        if step_histories:
            execution_context = f"""
            Previous attempts for current goal (including mapped steps from previous plans):
            """
            for history in step_histories:
                execution_context += f"""
                Step: {history.plan_step_description}
                Total attempts: {len(history.attempts)}
                Failed attempts: {len(history.all_failed_attempts)}
                
                Attempt history:
                """
                for i, attempt in enumerate(history.attempts, 1):
                    execution_context += f"""
                    Attempt {i}:
                    - Plan: {attempt.detailed_plan.plan_description}
                    - Actions taken: {', '.join(str(step) for step in attempt.detailed_plan.steps)}
                    - Technical success: {attempt.execution_success}
                    - Step completed: {attempt.step_completed}
                    - Reasoning: {attempt.evaluation_reasoning}
                    """

        prompt = f"""
        Evaluate the current progress of task execution.

        Original task goal: {current_context.user_task}
        Current step ({current_context.current_step + 1} of {len(current_context.current_plan.steps)}):  # Fixed this line
        {current_step.description}
        
        Step's expected outcome:
        {current_step.expected_outcome.description if current_step.expected_outcome else 'Unknown'}
        Expected elements: {', '.join(current_step.expected_outcome.key_elements) if current_step.expected_outcome else 'Unknown'}

        {execution_context}

        Current screen elements:
        {screen.elements}

        Evaluate:
        1. Has this specific step been completed based on its expected outcome?
        2. Does the current screen state match what we need to proceed?
        3. Most importantly - has the original task goal been achieved?

        Consider:
        - Previous failed attempts for this step and why they failed
        - Whether the technical success of actions actually achieved the desired outcome

        Format response as JSON:
        {{
            "step_completed": true/false,
            "reasoning": "Detailed explanation of the decision",
            "next_action": "proceed" or "replan",
            "success_criteria_met": ["list", "of", "matched criteria"],
            "task_goal_achieved": false,
            "failure_analysis": "If step not completed, analysis of why previous attempts failed"
        }}
        """

        try:
            response = self.planner_model.generate_content(prompt, screen.image, system_prompt=self.system_prompt)
            evaluation = Model.parse_json(response)

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