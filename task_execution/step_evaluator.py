from dataclasses import dataclass
from typing import Optional
from models.api import Model
from task_execution.data_structures import StepPlan
from vision.vision_analysis import ScreenAnalysis

@dataclass
class EvaluationResult:
    """Result of step plan execution evaluation"""
    success: bool
    justification: str
    confidence: float
    detected_changes: list[str]
    
class StepEvaluator:
    """Evaluates if step plan execution was successful by comparing screens"""

    def __init__(self, model: Model):
        self.model = model
        self.system_prompt = """You are an expert UI automation QA engineer.
Your role is to:
1. Carefully analyze screen states before and after step execution
2. Compare actual results against expected outcomes
3. Detect meaningful changes between screens
4. Determine if changes match the intended plan
5. Provide detailed justification for your evaluation
6. Consider both positive and negative indicators
7. Be thorough but focused on relevant changes

Frame your analysis in terms of:
- Visible changes that indicate success
- Missing changes that indicate failure
- Unexpected changes that need investigation
- Relationship between changes and expected outcome
"""

    def evaluate_progress(self, 
                         plan: StepPlan,
                         current_screen: ScreenAnalysis,
                         next_screen: ScreenAnalysis) -> EvaluationResult:
        """Evaluate if step plan execution was successful"""
        
        # Analyze both screens to ensure we have text analysis
        current_screen.analyze_text_screen()
        next_screen.analyze_text_screen()

        prompt = self._generate_evaluation_prompt(
            plan_description=plan.plan_description,
            expected_outcome=plan.expected_outcome, 
            before_summary=current_screen.text_screen_info.summary,
            before_description=current_screen.text_screen_info.screen_description,
            interface_type=current_screen.environment.interface_type.name
        )

        response = self.model.generate_content(
            prompt,
            next_screen.image,
            system_prompt=self.system_prompt
        )

        if not response:
            raise ValueError("Model returned null response")

        try:
            result = Model.parse_json(response)
            
            # Validate required fields
            required_fields = ['success', 'justification', 'confidence', 'detected_changes']
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                raise ValueError(f"Response missing required fields: {missing_fields}")

            return EvaluationResult(
                success=result['success'],
                justification=result['justification'],
                confidence=result['confidence'],
                detected_changes=result['detected_changes']
            )

        except Exception as e:
            print(f"Failed to parse evaluation response: {str(e)}")
            return EvaluationResult(
                success=False,
                justification="Failed to evaluate progress due to error",
                confidence=0.0,
                detected_changes=[]
            )

    def _generate_evaluation_prompt(self,
                                  plan_description: str,
                                  expected_outcome: str,
                                  before_summary: str,
                                  before_description: str,
                                  interface_type: str) -> str:
        """Generate the evaluation prompt"""
        return f"""
        Compare the previous screen state with the current screen and determine if the step plan was executed successfully.

        Plan Description: {plan_description}
        Expected Outcome: {expected_outcome}
        Interface Type: {interface_type}

        Previous Screen State:
        Summary: {before_summary}
        Description: {before_description}

        Analyze the provided screenshot of the current screen state and evaluate:
        1. Are the expected changes visible?
        2. Does the new state match the expected outcome?
        3. Are there any unexpected or missing changes?
        4. How confident are you in the evaluation?

        Return your analysis as JSON:
        {{
            "success": boolean,
            "justification": "Detailed explanation of your evaluation",
            "confidence": float between 0 and 1,
            "detected_changes": [
                "list of specific changes detected between screens"
            ]
        }}

        Guidelines:
        - Focus on changes relevant to the expected outcome
        - Consider both positive and negative indicators
        - Look for clear evidence of success/failure
        - Note any ambiguous or concerning changes
        - Be specific about what changed and why it matters
        - Consider the interface type when evaluating changes
        """
