from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass 
class PlanCondition:
    """Describes a system condition in natural language"""
    description: str  # Natural language description of required/expected state
    key_elements: List[str]  # List of key UI elements that should be present/visible

@dataclass
class PlanStep:
    """A high-level step in the execution plan"""
    description: str  # What needs to be done in natural language
    precondition: PlanCondition  # What should be true before this step
    expected_outcome: Optional[PlanCondition]  # What should be true after, None if unknown

@dataclass
class ExecutionPlan:
    """Complete plan for executing a task"""
    initial_state: PlanCondition
    steps: List[PlanStep]
    is_complete: bool 

    def to_dict(self) -> Dict:
        """Convert ExecutionPlan to a dictionary"""
        return {
            "initial_state": {
                "description": self.initial_state.description,
                "key_elements": self.initial_state.key_elements
            },
            "steps": [
                {
                    "description": step.description,
                    "precondition": {
                        "description": step.precondition.description,
                        "key_elements": step.precondition.key_elements
                    },
                    "expected_outcome": {
                        "description": step.expected_outcome.description,
                        "key_elements": step.expected_outcome.key_elements
                    } if step.expected_outcome else None
                }
                for step in self.steps
            ],
            "is_complete": self.is_complete
        }

    @staticmethod
    def from_dict(data: Dict) -> 'ExecutionPlan':
        """Create ExecutionPlan from a dictionary"""
        return ExecutionPlan(
            initial_state=PlanCondition(
                description=data["initial_state"]["description"],
                key_elements=data["initial_state"]["key_elements"]
            ),
            steps=[
                PlanStep(
                    description=step["description"],
                    precondition=PlanCondition(
                        description=step["precondition"]["description"],
                        key_elements=step["precondition"]["key_elements"]
                    ),
                    expected_outcome=PlanCondition(
                        description=step["expected_outcome"]["description"],
                        key_elements=step["expected_outcome"]["key_elements"]
                    ) if step["expected_outcome"] else None
                )
                for step in data["steps"]
            ],
            is_complete=data["is_complete"]
        )

    def print_plan(self):
        """Print the execution plan in a human-readable format"""
        print("Execution Plan:")
        print(f"Initial State: {self.initial_state.description}")
        print(f"Key Elements: {', '.join(self.initial_state.key_elements)}")
        print("\nSteps:")
        for i, step in enumerate(self.steps):
            print(f"Step {i + 1}:")
            print(f"  Description: {step.description}")
            print(f"  Precondition: {step.precondition.description}")
            print(f"  Key Elements: {', '.join(step.precondition.key_elements)}")
            if step.expected_outcome:
                print(f"  Expected Outcome: {step.expected_outcome.description}")
                print(f"  Key Elements: {', '.join(step.expected_outcome.key_elements)}")
            else:
                print("  Expected Outcome: Unknown")
            print()

@dataclass
class StepInstruction:
    """Represents a single UI interaction instruction"""
    action: str
    target: Optional[str] = None
    coordinates: Optional[tuple[int, int]] = None
    wait_time: Optional[float] = None
    success_condition: Optional[str] = None
    context: Optional[Dict] = None
    interaction_details: Optional[Dict] = None

@dataclass
class DetailedStep(StepInstruction):
    """Extends StepInstruction with additional planning details"""
    description: str = ""
    expected_result: Optional[str] = None

    def __str__(self):
        return (f"DetailedStep(description={self.description}, action={self.action}, "
                f"target={self.target}, coordinates={self.coordinates}, "
                f"context={self.context}, expected_result={self.expected_result})")

@dataclass
class TaskContext:
    """Maintains the context for task execution"""
    user_task: str
    plan: ExecutionPlan
    current_step: int
    current_phase: str
    observations: List[str]  # Recent relevant observations
    attempted_actions: List[str]  # History of actions tried

    def to_json(self, file_path: str):
        """Serialize TaskContext to a JSON file"""
        with open(file_path, 'w') as file:
            json.dump({
                'user_task': self.user_task,
                'plan': self.plan.to_dict(),
                'current_step': self.current_step,
                'current_phase': self.current_phase,
                'observations': self.observations,
                'attempted_actions': self.attempted_actions
            }, file, indent=4)

    @staticmethod
    def from_json(file_path: str) -> 'TaskContext':
        """Deserialize TaskContext from a JSON file"""
        with open(file_path, 'r') as file:
            data = json.load(file)
            plan_data = data['plan']
            plan = ExecutionPlan.from_dict(plan_data)
            return TaskContext(
                user_task=data['user_task'],
                plan=plan,
                current_step=data['current_step'],
                current_phase=data['current_phase'],
                observations=data['observations'],
                attempted_actions=data['attempted_actions']
            )

@dataclass
class StepPlan:
    """Contains the complete plan including steps and expected outcome"""
    steps: List[DetailedStep]
    plan_description: str 
    expected_outcome: str

