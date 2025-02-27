from dataclasses import dataclass, field
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
from fuzzywuzzy import fuzz # type: ignore
import torch
from lib.OmniParser.omni_utils import *

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
    input_methods: Set[str] = field(default_factory=set)

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

    BOX_TRESHOLD=0.01
    model_path = "lib/OmniParser/weights/icon_detect/model.pt"
    som_model = get_yolo_model(model_path)

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

    def __init__(self, image: Optional[Image.Image] = None, debug_mode: bool = False):
        
        self.debug_mode = debug_mode
        self.image= image

        self.model = Model.create_from_config("planning_model")
        self.provider = VisionLLMBase.create_from_config("googlevision")
        self.provider_coords = VisionLLMBase.create_from_config("qwenvision")
        self.provider_labels = Model.create_from_config("label_model")
        
        self.text_analysis: Optional[TextScreenAnalysis] = None
        self._is_text_analyzed = False
        self.ocr_bbox = None
        self.ocr_text = None
        self.ocr_bbox_elem = None
        self.filtered_boxes_elem = None
        
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
        self._is_yolo_ocr_processed = False

    def ensure_screen_added(self) -> 'ScreenAnalysis':
        if self.debug_mode:
            raise RuntimeError("Cannot add screen in debug mode")

        """Lazily add screen to providers"""
        assert self.provider is not None, "Vision provider not found"
        assert self.provider_coords is not None, "Vision provider for coordinates not found"
        assert self.image is not None, "No screenshot available for analysis"

        if not self._is_screen_added:
            self.provider.add_screen(self.image)
            self.provider_coords.add_screen(self.image)
            self._is_screen_added = True
            self._prepprocess_image()

        return self


    def _get_annotated_image(self, _elements, logits, phrases):
        image_source = np.asarray(self.image)
        annotated_frame, cropped_images = annotate_and_crop(image_source=image_source, boxes=torch.tensor(_elements), logits=logits, phrases=phrases)
        pil_img = Image.fromarray(annotated_frame)
        return pil_img, cropped_images[phrases[0]]
    

    def _prepprocess_image(self):

        if self._is_yolo_ocr_processed:
            return

        assert self.image is not None, "Image not found"

        w, h = self.image.size
        imgsz = (h, w)
        xyxy, logits, phrases = predict_yolo(model=self.som_model, image=self.image, box_threshold=self.BOX_TRESHOLD, imgsz=imgsz, scale_img=False, iou_threshold=0.1)
        # xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
        phrases = [str(i) for i in range(len(phrases))]

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(self.image, display_img = False, output_bb_format='xyxy', goal_filtering=None)
        self.ocr_text, self.ocr_bbox = ocr_bbox_rslt
        self.ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(self.ocr_bbox, self.ocr_text) if int_box_area(box, w, h) > 0] 

        iou_threshold=0.9
        xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
        
        #print(" x" * 80)
        # print(" >> len yolo", len(xyxy))
        # print(" >< len ocr", len(self.ocr_bbox))
        # print("combined", len(xyxy_elem))
        # print(" x" * 80)
        filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=self.ocr_bbox_elem)

        # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
        self.filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
        # print("filtered_boxes_elem", self.filtered_boxes_elem)
        filtered_boxes = torch.tensor([box['bbox'] for box in self.filtered_boxes_elem])
        filtered_boxes = filtered_boxes / torch.Tensor([w, h, w, h]).to(filtered_boxes.device)
        for i, b in enumerate(self.filtered_boxes_elem):
            if b["source"] == "box_yolo_content_yolo": 
                _l = []
                _l.append(filtered_boxes[i].tolist())
                _i, _i2 = self._get_annotated_image(_l, logits, [f"{i}"])
                #_i2 = get_cropped_image(filtered_boxes[i], image_source)
                result = self.provider_labels.generate_content(f"you are provided with 2 images. label the element with number '0' in a single sentence. there is an additional cropped image just containing the content of the bounding box.\n\n Important: Output only the element type, the label if avalable, and its function. \n Example: folder icon, named 'Works', representing the current directory", [_i, _i2]) 
                print(i, result)
                self.filtered_boxes_elem[i]["content"] = result
                self._is_yolo_ocr_processed = True

        # print(self.filtered_boxes_elem)
        # get the index of the first 'content': None
        # starting_idx = next((i for i, box in enumerate(self.filtered_boxes_elem) if box['content'] is None), -1)
        # self.filtered_boxes = torch.tensor([box['bbox'] for box in self.filtered_boxes_elem])
        

    def analyze_text_screen(self) -> 'ScreenAnalysis':
        """Perform OCR and LLM analysis for all interface types"""
        
        self.detect_environment()
        
        if not self._is_text_analyzed:
            if self.debug_mode:
                raise RuntimeError("Cannot analyze text screen in debug mode")
        
            assert self.image is not None, "No screenshot available for analysis"
            assert self.environment is not None, "Environment not detected"
            assert self.model is not None, "Model not found"
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

        assert self.model is not None, "Model not found"
        assert self.image is not None, "No screenshot available for analysis"
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
        
        self.ensure_screen_added()
        self.detect_environment()
        
        if not self._is_elements_detected:
            if self.debug_mode:
                raise RuntimeError("Cannot detect elements in debug mode")
        
            self.analyze_text_screen()
            assert self.provider is not None, "Vision provider not found"   
            assert self.environment is not None, "Environment not detected"
            assert self.capabilities is not None, "Capabilities not detected"
            assert self.required_analysis is not None, "Required analysis not detected"
            if self.environment.interface_type in [InterfaceType.TERMINAL, InterfaceType.TEXT_UI]:
                # For text-based interfaces, analyze text first
                
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
           
            assert self.filtered_boxes_elem is not None, "No filtered boxes found"
            # Filter box detection results
            box_elements = [elem for elem in self.filtered_boxes_elem]
            # Process each box element
            for box in box_elements:
                matched = False
                bbox = box['bbox']
                
                # Convert bbox to standardized dict format
                bbox_dict = {
                    "y1": bbox[1] if isinstance(bbox, tuple) else bbox[1], 
                    "x1": bbox[0] if isinstance(bbox, tuple) else bbox[0],
                    "y2": bbox[3] if isinstance(bbox, tuple) else bbox[3], 
                    "x2": bbox[2] if isinstance(bbox, tuple) else bbox[2]
                }
                
                # Try to match with existing elements using fuzzy matching
                if box.get('content'):
                    best_match_score = 0
                    best_match_element = None
                    
                    for element in self.ui_elements:
                        if not element.text:
                            continue
                            
                        # Calculate fuzzy match ratio
                        ratio = fuzz.ratio(box['content'].lower(), element.text.lower())
                        
                        # Also check partial token matches for longer strings
                        token_ratio = fuzz.partial_token_sort_ratio(box['content'].lower(), element.text.lower())
                        
                        # Use the higher of the two scores
                        match_score = max(ratio, token_ratio)
                        if match_score > best_match_score and match_score > 70:  # 70% threshold
                            best_match_score = match_score
                            best_match_element = element
                    
                    if best_match_element:
                        best_match_element.bounding_box = bbox_dict
                        matched = True
                
                # Add new element if no match found
                if not matched:
                    element_type = "button" if box['interactivity'] else "text"
                    context = ""
                    
                    new_element = {
                        "element_id": f"{box['content'].lower().replace(' ', '_') if box['content'] else 'icon'}_{len(self.ui_elements)}",
                        "element_type": element_type,
                        "element_text": box['content'] or "",
                        "context": context,
                        "visual_description": f"{element_type.capitalize()} containing '{box['content']}'" if box['content'] else "Icon element",
                        "importance": "high" if box['interactivity'] else "medium",
                        "bounding_box": bbox_dict
                    }
                    
                    self.ui_elements.append(UIElement.from_dict(new_element))
            
            # print(f"Detected UI elements: {self.ui_elements}")

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
        assert self.provider_coords is not None, "Vision provider for coordinates not found"
        return self.provider_coords.get_click_coordinates_by_description(description)

    def _find_yolo_ocr_coordinates(self, elem: UIElement) -> Optional[dict]:
        """Find coordinates from YOLO and OCR results"""
        if self.ocr_bbox_elem and elem.text:
            print(f"Detecting bounding box for {elem.text}")
            for ocr_elem in self.ocr_bbox_elem:
                if ocr_elem['content'] == elem.text:
                    # Convert from xyxy format to the required dict format
                    bbox = ocr_elem['bbox']
                    abs_x1 = bbox[0]
                    abs_y1 = bbox[1]
                    abs_x2 = bbox[2]
                    abs_y2 = bbox[3]                      

                    center_x = (abs_x1 + abs_x2) // 2
                    center_y = (abs_y1 + abs_y2) // 2
    
                    coords =  {"x": center_x, "y" : center_y}    
                    return coords
        return None

    def get_coordinates(self, elem: UIElement, update: bool = False) -> dict:
        """Lazily add click coordinates if needed"""

        if self.debug_mode:
            return { "x" : "0", "y" : "0" } 
        self.detect_elements()

        print(f"Getting coordinates for {elem}")
        result = self._find_yolo_ocr_coordinates(elem)
        if result:
            return result
        assert self.capabilities is not None, "Capabilities not detected"
        assert self.provider_coords is not None, "Vision provider for coordinates not found"
        if self.capabilities.has_mouse:
            coords = self.provider_coords.get_click_coordinates(elem)
            elem.click_coordinates = coords
        else:
            raise ValueError("Mouse input not supported")

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
        if not value:
            return
        self.ui_elements = value
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

    @property
    def orc_text(self) -> Optional[str]:
        """Get OCR text if available"""
        if not self.ocr_text:
            self._prepprocess_image()
        result_text = " ".join(self.ocr_text) if self.ocr_text else ""
        return result_text

    def save_debug_info(self, plan) -> Path:
        """Save debug information to a timestamped directory."""
        # Create base debug directory
        assert self.image is not None, "No screenshot available for debug"
        
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
    