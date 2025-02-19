from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import time
import uuid


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
        details = []
        details.append(f"Action: {self.action}")
        if self.description:
            details.append(f"Description: {self.description}")
        if self.target:
            details.append(f"Target: {self.target}")
        if self.coordinates:
            details.append(f"Coordinates: {self.coordinates}")
        if self.context:
            if 'input_text' in self.context:
                details.append(f"Input: {self.context['input_text']}")
            if 'additional_notes' in self.context:
                details.append(f"Notes: {self.context['additional_notes']}")
        if self.expected_result:
            details.append(f"Expected: {self.expected_result}")
        return " | ".join(details)
   

@dataclass
class StepPlan:
    """Contains the complete plan including steps and expected outcome"""
    steps: List[DetailedStep]
    plan_description: str 
    expected_outcome: str

from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import time

@dataclass
class StepAttempt:
    """Records an attempt to execute a high-level plan step"""
    timestamp: float
    detailed_plan: StepPlan  # The detailed steps that were attempted
    execution_success: bool  # Whether the detailed steps executed without errors
    step_completed: bool  # Whether the high-level plan step was achieved
    evaluation_reasoning: str  # Why the attempt succeeded/failed
    screen_state: Dict  # UI state at time of evaluation
    failure_details: Optional[str] = None
    interaction_result: Optional[str] = None

    @staticmethod
    def create_new(
        detailed_plan: StepPlan,
        execution_success: bool,
        step_completed: bool,
        evaluation_reasoning: str,
        screen_state: Dict
    ) -> 'StepAttempt':
        return StepAttempt(
            timestamp=time.time(),
            detailed_plan=detailed_plan,
            execution_success=execution_success,
            step_completed=step_completed,
            evaluation_reasoning=evaluation_reasoning,
            screen_state=screen_state
        )

    def get_screen_summary(self) -> str:
        """Create a concise summary of the screen state"""
        if not self.screen_state or 'elements' not in self.screen_state:
            return "No screen state recorded"
        
        elements = self.screen_state['elements']
        summary = []
        for element in elements[:5]:  # Show first 5 elements
            if isinstance(element, dict):
                summary.append(f"- {element.get('type', 'Unknown')}: {element.get('text', 'No text')}")
            else:
                summary.append(f"- {str(element)}")
        if len(elements) > 5:
            summary.append(f"... and {len(elements) - 5} more elements")
        return "\n".join(summary)

@dataclass
class PlanStepHistory:
    """Tracks all attempts for a specific step in the plan"""
    plan_step_index: int
    plan_step_description: str
    attempts: List[StepAttempt]

    def add_attempt(self, attempt: StepAttempt):
        self.attempts.append(attempt)

    @property
    def latest_attempt(self) -> Optional[StepAttempt]:
        return self.attempts[-1] if self.attempts else None

    @property
    def all_failed_attempts(self) -> List[StepAttempt]:
        return [a for a in self.attempts if not a.step_completed]

    def to_dict(self) -> Dict:
        return {
            'plan_step_index': self.plan_step_index,
            'plan_step_description': self.plan_step_description,
            'attempts': [
                {
                    'timestamp': attempt.timestamp,
                    'detailed_plan': {
                        'steps': [vars(step) for step in attempt.detailed_plan.steps],
                        'plan_description': attempt.detailed_plan.plan_description,
                        'expected_outcome': attempt.detailed_plan.expected_outcome
                    },
                    'execution_success': attempt.execution_success,
                    'step_completed': attempt.step_completed,
                    'evaluation_reasoning': attempt.evaluation_reasoning,
                    'screen_state': attempt.screen_state
                }
                for attempt in self.attempts
            ]
        }

@dataclass
class PlanVersion:
    """Represents a version of the execution plan"""
    plan: ExecutionPlan
    created_at: float
    reason: str
    step_mapping: Optional[Dict[int, int]] = None  # Maps steps from previous version to this one

@dataclass
class TaskContext:
    """Maintains the context for task execution"""
    user_task: str
    plan_versions: List[PlanVersion]  # History of all plans
    current_step: int
    current_phase: str
    observations: List[str]
    step_histories: Dict[str, PlanStepHistory]  # Maps "plan_version:step_index" to history
    
    @property
    def current_plan(self) -> Optional[ExecutionPlan]:
        """Get the current active plan"""
        return self.plan_versions[-1].plan if self.plan_versions else None

    def add_new_plan(self, new_plan: ExecutionPlan, reason: str, step_mapping: Optional[Dict[int, int]] = None):
        """
        Add a new plan version with optional mapping of steps from previous plan
        
        Args:
            new_plan: The new execution plan
            reason: Why replanning was needed
            step_mapping: Dictionary mapping steps from old plan to new plan
                        e.g., {0: 1, 2: 3} means old step 0 corresponds to new step 1
        """
        version = PlanVersion(
            plan=new_plan,
            created_at=time.time(),
            reason=reason,
            step_mapping=step_mapping
        )
        self.plan_versions.append(version)

    def get_step_key(self, step_index: int) -> str:
        """Generate unique key for a step in current plan"""
        plan_version = len(self.plan_versions) - 1
        return f"{plan_version}:{step_index}"

    
    def get_step_history(self, step_index: int, include_mapped: bool = True) -> List[PlanStepHistory]:
        """
        Get the complete history for a step, including mapped steps from previous plans
        
        Args:
            step_index: Current step index
            include_mapped: Whether to include history from mapped steps in previous plans
        """
        histories = []
        
        # Get current plan's history
        current_key = self.get_step_key(step_index)
        if current_key in self.step_histories:
            histories.append(self.step_histories[current_key])
        
        if include_mapped:
            # Look for mapped steps in previous plans
            for version_idx in range(len(self.plan_versions) - 1, -1, -1):
                version = self.plan_versions[version_idx]
                if version.step_mapping:
                    # Find if current step maps to any step in this version
                    for old_idx, new_idx in version.step_mapping.items():
                        if new_idx == step_index:
                            old_key = f"{version_idx}:{old_idx}"
                            if old_key in self.step_histories:
                                histories.append(self.step_histories[old_key])
        
        return histories
    
    def record_attempt(self, 
                      step_index: int,
                      attempt: StepAttempt):
        """Record a new attempt for a plan step"""
        step_key = self.get_step_key(step_index)
        
        if step_key not in self.step_histories:
            self.step_histories[step_key] = PlanStepHistory(
                plan_step_index=step_index,
                plan_step_description=self.current_plan.steps[step_index].description if self.current_plan else "",
                attempts=[]
            )
        
        self.step_histories[step_key].add_attempt(attempt)

    def get_current_step_history(self, include_mapped: bool = True) -> List[PlanStepHistory]:
        """Get the history for the current step, including mapped steps"""
        return self.get_step_history(self.current_step, include_mapped)

    def to_json(self, file_path: str):
        """Serialize TaskContext to a JSON file"""
        with open(file_path, 'w') as file:
            json.dump({
                'user_task': self.user_task,
                'plan_versions': [
                    {
                        'plan': version.plan.to_dict(),
                        'created_at': version.created_at,
                        'reason': version.reason,
                        'step_mapping': version.step_mapping
                    }
                    for version in self.plan_versions
                ],
                'current_step': self.current_step,
                'current_phase': self.current_phase,
                'observations': self.observations,
                'step_histories': {
                    key: history.to_dict()
                    for key, history in self.step_histories.items()
                }
            }, file, indent=4)

    def create_execution_summary(self) -> str:
        summary = []
        summary.append("=== Task Execution Summary ===")
        summary.append(f"Task Goal: {self.user_task}")
        summary.append(f"Total Plan Versions: {len(self.plan_versions)}\n")
        
        for version_idx, plan_version in enumerate(self.plan_versions):
            summary.append(f"\nPlan Version {version_idx + 1}")
            summary.append("="* 50)
            summary.append(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(plan_version.created_at))}")
            if version_idx > 0:
                summary.append(f"Reason for replanning:\n{plan_version.reason}")
            
            for step_idx, step in enumerate(plan_version.plan.steps):
                summary.append(f"\nStep {step_idx + 1}: {step.description}")
                summary.append("-" * 40)
                summary.append(f"Precondition: {step.precondition.description}")
                summary.append(f"Required elements: {', '.join(step.precondition.key_elements)}")
                summary.append(f"Expected outcome: {step.expected_outcome.description if step.expected_outcome else 'Unknown'}")
                
                step_key = f"{version_idx}:{step_idx}"
                if step_key in self.step_histories:
                    history = self.step_histories[step_key]
                    summary.append(f"\nExecution Attempts ({len(history.attempts)}):")
                    
                    for attempt_idx, attempt in enumerate(history.attempts, 1):
                        summary.append(f"\n  Attempt {attempt_idx}:")
                        summary.append(f"  Time: {time.strftime('%H:%M:%S', time.localtime(attempt.timestamp))}")
                        summary.append("  Actions:")
                        for detailed_step in attempt.detailed_plan.steps:
                            summary.append(f"    {str(detailed_step)}")
                        
                        summary.append(f"  Technical Success: {'✓' if attempt.execution_success else '✗'}")
                        summary.append(f"  Step Completed: {'✓' if attempt.step_completed else '✗'}")
                        summary.append(f"  Evaluation:\n    {attempt.evaluation_reasoning}")
                        
                        summary.append("  Screen State:")
                        summary.append(f"    {attempt.get_screen_summary()}")
                else:
                    summary.append("\nNo execution attempts recorded")
                
                summary.append("-" * 40)
    
        summary.append("\nFinal Status")
        summary.append("=" * 50)
        summary.append(f"Current plan version: {len(self.plan_versions)}")
        summary.append(f"Current step: {self.current_step + 1}")
        summary.append(f"Current phase: {self.current_phase}")
        
        return "\n".join(summary)

    def save_execution_summary(self, file_path: str):
        """Save the execution summary to a file"""
        with open(file_path, 'w') as f:
            f.write(self.create_execution_summary())