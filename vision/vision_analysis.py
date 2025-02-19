from dataclasses import dataclass
from typing import List, Optional, Set
import json
from pathlib import Path
from enum import Enum, auto
from PIL import Image
from vision.vision_base import UIElement, VisionLLMBase
from models.api import Model
import datetime
import json
from pathlib import Path


class InterfaceType(Enum):
    TERMINAL = auto()
    TEXT_UI = auto()
    GUI = auto()
    WEB = auto()
    UNKNOWN = auto()

@dataclass
class RequiredAnalysis:
    needs_ocr: bool
    needs_ui_detection: bool

@dataclass
class SoftwareEnvironment:
    os: str
    interface_type: InterfaceType
    window_system: Optional[str] = None
    terminal_type: Optional[str] = None
    application: Optional[str] = None

@dataclass
class InteractionCapabilities:
    has_keyboard: bool = True
    has_mouse: bool = False
    has_touch: bool = False
    supports_text_selection: bool = False
    supports_clipboard: bool = False
    input_methods: Set[str] = None

@dataclass
class TextScreenInteraction:
    keys: List[str]
    description: str
    location: str  # Where on the screen this option is shown

@dataclass
class TextScreenAnalysis:
    summary: str          # General context of what we're looking at
    screen_description: str  # What is visible on the screen
    visible_interactions: List[TextScreenInteraction]  # Only explicitly shown options


class ScreenAnalysis:
    SYSTEM_PROMPT = """You are an expert in analyzing computer interfaces and operating systems.
Your task is to examine the provided screenshot and identify:
1. The operating system and its characteristics
2. The type of interface (terminal, GUI, web, etc.)
3. Any visible applications or window systems
4. Available interaction methods

Provide your analysis in valid JSON format with the following structure:
{
    "os": "detailed os name and version if visible",
    "interface_type": "TERMINAL|TEXT_UI|GUI|WEB|UNKNOWN",
    "window_system": "window system name if applicable",
    "terminal_type": "terminal type if applicable",
    "application": "visible application name if any",
    "characteristics": {
        "has_mouse_support": boolean,
        "has_touch_support": boolean,
        "supports_text_selection": boolean,
        "supports_clipboard": boolean
    },
    "available_input_methods": ["list", "of", "possible", "input", "methods"],
    "confidence": float between 0 and 1
}"""

    ANALYSIS_PROMPT = """Analyze this screenshot and identify the operating environment and interface characteristics.
Focus on:
- Visual elements that indicate the operating system
- Interface type (terminal vs GUI)
- Visible applications or window systems
- Available interaction methods
- Any unique characteristics that might affect interaction

Provide your complete analysis in the required JSON format."""

    TEXT_SCREEN_PROMPT = """Analyze this terminal/text screen and provide DETAILED information about:
1. A brief summary of the context (what kind of screen/application we're looking at)
2. A detailed description of what is actually visible on the screen
3. A comprehensive list of ALL interaction options that are explicitly shown/visible on the screen

Format your response as JSON:
{
    "summary": "Brief context of what this screen represents",
    "screen_description": "Detailed description of what is visible on the screen",
    "visible_interactions": [
        {
            "keys": ["exact", "keys", "shown"],
            "description": "The exact action description as shown on screen",
            "location": "Precise location where this option is shown (e.g., 'bottom status line', 'top menu bar', 'line 5')"
        }
    ]
}

IMPORTANT:
- Be extremely specific about key combinations (e.g., "Ctrl+C", "ESC", "F1")
- Include ALL visible key bindings or command options shown on screen
- Note the exact location of each interaction option
- Include menu options, status line commands, and visible shortcuts
- For function keys, use format "F1", "F2", etc.
- For special keys, use proper names: "ESC", "ENTER", "SPACE"
- For combinations, use format "Ctrl+X", "Alt+F", etc.
- Do not infer or suggest possible commands - only include what is explicitly shown
- Pay special attention to:
  * Command menus or help lines at top/bottom of screen
  * Status bars with key information
  * Dialog boxes with button options
  * Any visible key bindings or shortcuts
- Include exact text of commands and options as shown on screen"""

    GUI_SCREEN_PROMPT = """Analyze this GUI/web interface screenshot and provide:
1. A brief contextual summary of what application/system we're looking at
2. A high-level description of the available user interaction options

Format your response as JSON:
{
    "summary": "Brief context of what this interface represents",
    "interaction_overview": "High-level description of how users can interact with this interface",
    "primary_actions": ["list", "of", "main", "possible", "actions"]
}

IMPORTANT:
- Focus on the main purpose and primary interaction patterns
- Keep descriptions concise and user-focused
- Include only clearly visible/available options
- Describe the interface from a user's perspective"""

    def __init__(self, 
                 image: Optional[Image.Image] = None, 
                 model: Optional[Model] = None, 
                 provider: Optional[VisionLLMBase] = None, 
                 provider_coords: Optional[VisionLLMBase] = None):
        
        if not image and not model and not provider and not provider_coords:
            self.debug_mode = True
        else:
            self.debug_mode = False

        self.image= image
        self.provider = provider
        self.provider_coords = provider_coords
        self.model = model
        self.text_analysis: Optional[TextScreenAnalysis] = None
        self._is_text_analyzed = False
        
        # Environment analysis
        self.environment: Optional[SoftwareEnvironment] = None
        self.capabilities: Optional[InteractionCapabilities] = None
        self.required_analysis: Optional[RequiredAnalysis] = None
        self._analysis_confidence: float = 0.0
        
        # UI elements
        self.ui_elements: List[UIElement] = []
        self._elements_data = None
        
        # State tracking
        self._is_screen_added = False
        self._is_environment_detected = False
        self._is_elements_detected = False
        self._is_coordinates_added = False

    def ensure_screen_added(self) -> 'ScreenAnalysis':
        if self.debug_mode:
            raise RuntimeError("Cannot add screen in debug mode")

        """Lazily add screen to providers"""
        if not self._is_screen_added:
            self.provider.add_screen(self.image)
            self.provider_coords.add_screen(self.image)
            self._is_screen_added = True
        return self

    def analyze_text_screen(self) -> 'ScreenAnalysis':
        """Perform OCR and LLM analysis for all interface types"""
        
        self.detect_environment()
        
        if not self._is_text_analyzed:
            if self.debug_mode:
                raise RuntimeError("Cannot analyze text screen in debug mode")
        
            try:
                img = self.image
                if self.environment.interface_type in [InterfaceType.TERMINAL, InterfaceType.TEXT_UI]:
                    # Original text-based analysis
                    text_analysis = self.model.generate_content(
                        self.TEXT_SCREEN_PROMPT,
                        img,
                        system_prompt="You are an expert in analyzing terminal and text-based interfaces."
                    )
                    
                    analysis = Model.parse_json(text_analysis)
                    
                    self.text_analysis = TextScreenAnalysis(
                        summary=analysis["summary"],
                        screen_description=analysis["screen_description"],
                        visible_interactions=[
                            TextScreenInteraction(
                                keys=interaction["keys"],
                                description=interaction["description"],
                                location=interaction["location"]
                            )
                            for interaction in analysis["visible_interactions"]
                        ]
                    )
                else:
                    # GUI/Web interface analysis
                    gui_analysis = self.model.generate_content(
                        self.GUI_SCREEN_PROMPT,
                        img,
                        system_prompt="You are an expert in analyzing graphical user interfaces and web applications."
                    )
                    
                    analysis = Model.parse_json(gui_analysis)
                    
                    self.text_analysis = TextScreenAnalysis(
                        summary=analysis["summary"],
                        screen_description=analysis["interaction_overview"],
                        visible_interactions=[
                            TextScreenInteraction(
                                keys=[],  # No specific keys for GUI actions
                                description=action,
                                location="interface"  # Generic location
                            )
                            for action in analysis["primary_actions"]
                        ]
                    )
                
                self._is_text_analyzed = True
                
            except (json.JSONDecodeError, KeyError) as e:
                raise RuntimeError(f"Failed to parse text screen analysis: {str(e)}")
        
        return self


    def detect_environment(self) -> 'ScreenAnalysis':
        """Detect the operating system and interface type using LLM analysis"""
        if self.debug_mode and self._is_environment_detected:
            return self
        elif self.debug_mode:      
            raise RuntimeError("Cannot detect environment in debug mode")

        self.ensure_screen_added()
        img = self.image
        if not self._is_environment_detected:
            llm_analysis = self.model.generate_content(
                self.ANALYSIS_PROMPT,
                img,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            try:
                analysis = Model.parse_json(llm_analysis)
                
                self.environment = SoftwareEnvironment(
                    os=analysis["os"],
                    interface_type=InterfaceType[analysis["interface_type"]],
                    window_system=analysis.get("window_system"),
                    terminal_type=analysis.get("terminal_type"),
                    application=analysis.get("application")
                )
                
                chars = analysis["characteristics"]
                self.capabilities = InteractionCapabilities(
                    has_keyboard=True,
                    has_mouse=chars["has_mouse_support"],
                    has_touch=chars["has_touch_support"],
                    supports_text_selection=chars["supports_text_selection"],
                    supports_clipboard=chars["supports_clipboard"],
                    input_methods=set(analysis["available_input_methods"])
                )
                
                self.required_analysis = self._determine_required_analysis(
                    self.environment,
                    self.capabilities
                )
                
                self._analysis_confidence = analysis["confidence"]
                self._is_environment_detected = True
                
            except (json.JSONDecodeError, KeyError) as e:
                raise RuntimeError(f"Failed to parse LLM analysis: {str(e)}")
        
        return self
    
  
    def detect_elements(self) -> 'ScreenAnalysis':
        """Lazily detect UI elements based on environment analysis"""
        
        self.detect_environment()
        
        if not self._is_elements_detected:
            if self.debug_mode:
                raise RuntimeError("Cannot detect elements in debug mode")
        
            if self.environment.interface_type in [InterfaceType.TERMINAL, InterfaceType.TEXT_UI]:
                # For text-based interfaces, analyze text first
                self.analyze_text_screen()
                # Convert text interactions to UI elements
                self._elements_data = []
                if self.text_analysis:
                    for i, interaction in enumerate(self.text_analysis.visible_interactions):
                        # Format keys for better readability
                        key_str = ", ".join(interaction.keys) if interaction.keys else "No keys"
                        
                        # Create a rich text description that includes all important information
                        element_text = (
                            f"{interaction.description}\n"
                            f"Keys: {key_str}\n"
                            f"Location: {interaction.location}"
                        )
                        
                        self._elements_data.append({
                            "id": f"interaction_{i}",
                            "type": "text_interaction",
                            "text": element_text,  # Combined information in text field
                            "description": interaction.description,  # Original action description
                            "context": f"{self.text_analysis.summary} | Keys: {key_str}",  # Include key info in context
                            "attributes": {  # Use attributes for structured data
                                "keys": key_str,
                                "location": interaction.location
                            }
                        })
            elif self.required_analysis.needs_ui_detection:
                self._elements_data = self.provider.detect_ui_elements()
            else:
                self._elements_data = []
                
            self.ui_elements = [UIElement.from_dict(elem) for elem in self._elements_data]
            self._is_elements_detected = True
        
        return self
    
    @property
    def text_screen_info(self) -> Optional[TextScreenAnalysis]:
        """Get text screen analysis if available"""
        if not self._is_text_analyzed: 
            self.analyze_text_screen()
        return self.text_analysis
    
    @text_screen_info.setter
    def text_screen_info(self, value: Optional[TextScreenAnalysis]):
        """Set text screen analysis"""
        print(f"Setting text screen info: {value}")
        self.text_analysis = value
        self._is_text_analyzed = True

    def get_coordinates_by_description(self, description: str, update: bool = False) -> dict:
        return self.provider_coords.get_click_coordinates_by_description(description)

    def get_coordinates(self, elem: UIElement, update: bool = False) -> dict:
        """Lazily add click coordinates if needed"""

        if self.debug_mode:
            return { "x" : "0", "y" : "0" } 
            
        self.detect_elements()

        print(f"Getting coordinates for {elem}")

        if not update and elem.click_coordinates is not None:
            return elem.click_coordinates

        if self.capabilities.has_mouse:
            coords = self.provider_coords.get_click_coordinates(elem)
            elem.click_coordinates = coords

        return elem.click_coordinates
    
    def _determine_required_analysis(
        self, 
        env: SoftwareEnvironment,
        caps: InteractionCapabilities
    ) -> RequiredAnalysis:
        """Determine what types of analysis are needed"""
        if env.interface_type == InterfaceType.TERMINAL:
            return RequiredAnalysis(
                needs_ocr=True,
                needs_ui_detection=False,
            )
        elif env.interface_type in [InterfaceType.GUI, InterfaceType.WEB]:
            return RequiredAnalysis(
                needs_ocr=False,
                needs_ui_detection=True,
            )
        return RequiredAnalysis(
            needs_ocr=True,
            needs_ui_detection=True,
        )

    @property
    def elements(self) -> List[UIElement]:
        """Get the currently detected UI elements"""
        if not self._is_elements_detected:
            self.detect_elements()
        return self.ui_elements
    
    @elements.setter
    def elements(self, value: List[UIElement]):
        """Set the detected UI elements"""
        self.ui_elements = [UIElement.from_dict(elem) for elem in value]
        self._is_elements_detected = True

    
    @property
    def analysis_strategy(self) -> dict:
        """Get the recommended analysis strategy"""
        if not self._is_environment_detected:
            self.detect_environment()
            
        return {
            "environment": self.environment,
            "capabilities": self.capabilities,
            "required_analysis": self.required_analysis,
            "confidence": self._analysis_confidence
        }
    
    @analysis_strategy.setter
    def analysis_strategy(self, value: dict):
        """Set the recommended analysis strategy"""
        self.environment = SoftwareEnvironment(
            os=value["environment"]["os"],
            interface_type=InterfaceType[value["environment"]["interface_type"]],
            window_system=value["environment"]["window_system"],
            terminal_type=value["environment"]["terminal_type"],
            application=value["environment"]["application"]
        )
        
        self.capabilities = InteractionCapabilities(
            has_keyboard=value["capabilities"]["has_keyboard"],
            has_mouse=value["capabilities"]["has_mouse"],
            has_touch=value["capabilities"]["has_touch"],
            supports_text_selection=value["capabilities"]["supports_text_selection"],
            supports_clipboard=value["capabilities"]["supports_clipboard"],
            input_methods=set(value["capabilities"]["input_methods"])
        )
        
        self.required_analysis = RequiredAnalysis(
            needs_ocr=value["required_analysis"]["needs_ocr"],
            needs_ui_detection=value["required_analysis"]["needs_ui_detection"]
        )
        
        self._analysis_confidence = value["confidence"]
        self._is_environment_detected = True

    
    def save_debug_info(self, plan) -> Path:
        """Save debug information to a timestamped directory."""
        # Create base debug directory
        debug_dir = Path("vision-debug")
        debug_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory (e.g. "20240215-123456")
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_session = debug_dir / timestamp
        debug_session.mkdir()
        
        # Save screenshot
        screenshot_path = debug_session / "screenshot.png"
        self.image.save(screenshot_path)
        
        # Save environment analysis if available
        if self._is_environment_detected and self.environment and self.capabilities and self.required_analysis:
            env_data = {
                "environment": {
                    "os": self.environment.os,
                    "interface_type": self.environment.interface_type.name,
                    "window_system": self.environment.window_system,
                    "terminal_type": self.environment.terminal_type,
                    "application": self.environment.application
                },
                "capabilities": {
                    "has_keyboard": self.capabilities.has_keyboard,
                    "has_mouse": self.capabilities.has_mouse, 
                    "has_touch": self.capabilities.has_touch,
                    "supports_text_selection": self.capabilities.supports_text_selection,
                    "supports_clipboard": self.capabilities.supports_clipboard,
                    "input_methods": list(self.capabilities.input_methods) if self.capabilities.input_methods else []
                },
                "required_analysis": {
                    "needs_ocr": self.required_analysis.needs_ocr,
                    "needs_ui_detection": self.required_analysis.needs_ui_detection
                },
                "confidence": self._analysis_confidence
            }
            with open(debug_session / "environment.json", "w") as f:
                json.dump(env_data, f, indent=2)
                
        # Save text analysis if available
        if self._is_text_analyzed and self.text_analysis:
            text_data = {
                "summary": self.text_analysis.summary,
                "screen_description": self.text_analysis.screen_description,
                "visible_interactions": [
                    {
                        "keys": interaction.keys,
                        "description": interaction.description,
                        "location": interaction.location
                    }
                    for interaction in self.text_analysis.visible_interactions
                ]
            }
            with open(debug_session / "text_analysis.json", "w") as f:
                json.dump(text_data, f, indent=2)
                
        # Save UI elements if available
        if self._is_elements_detected:
            with open(debug_session / "ui_elements.json", "w") as f:
                json.dump(self._elements_data, f, indent=2)

        # Save step plan if provided
        if plan:
            plan_data = {
                "plan_description": plan.plan_description,
                "expected_outcome": plan.expected_outcome,
                "steps": [
                    {
                        "description": step.description,
                        "action": step.action,
                        "target": step.target,
                        "coordinates": step.coordinates,
                        "wait_time": step.wait_time,
                        "success_condition": step.success_condition,
                        "context": step.context,
                        "interaction_details": step.interaction_details,
                        "expected_result": step.expected_result
                    }
                    for step in plan.steps
                ]
            }
            with open(debug_session / "step_plan.json", "w") as f:
                json.dump(plan_data, f, indent=2)
                
                
        return debug_session
    