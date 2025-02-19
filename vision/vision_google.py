from vision.vision_base import VisionLLMBase, UIElement
from typing import List, Dict
from PIL import Image
from io import BytesIO
import json
from models.google_genai import GoogleModel
from vision.vision_google_constants import UI_DETECT_PROMPT, UI_ACTION_SCHEMA, BOUNDING_BOX_SYSTEM_INSTRUCTIONS
import os

class GoogleVision(VisionLLMBase):
    def __init__(self, apikey: str, scaled_image_size=1024):
        super().__init__()
        self.model = GoogleModel(apikey=apikey, system_prompt="You are a computer vision assistant specialized in analyzing user interfaces.")
        self.ui_detect_prompt = UI_DETECT_PROMPT
        self.action_schema = UI_ACTION_SCHEMA
        self.bounding_box_system_instruction = BOUNDING_BOX_SYSTEM_INSTRUCTIONS
        self.sclaled_image_size = scaled_image_size
        
        self.scale_x = 1.0
        self.scale_y = 1.0 

    def add_screen(self, image: Image.Image, override_cache: bool = False):
        super().add_screen(image, override_cache)
        self._prepare_image()

    def _parse_response(self, response_text: str) -> List[Dict]:
        try:
            response_data = json.loads(response_text)
            if not isinstance(response_data, dict) or 'ui_elements' not in response_data:
                print(response_text)
                raise ValueError("Response doesn't contain 'ui_elements' array")
                
            ui_elements = response_data['ui_elements']
            if not isinstance(ui_elements, list):
                print(response_text)
                raise ValueError("'ui_elements' is not an array")
                
            # Validate each element has required fields
            required_fields = ['element_id', 'element_type', 'element_text', 
                             'context', 'visual_description', 'importance']
            
            for element in ui_elements:
                missing_fields = [field for field in required_fields if field not in element]
                if missing_fields:
                    print(response_text)
                    raise ValueError(f"Element missing required fields: {missing_fields}")
                    
            return ui_elements
            
        except json.JSONDecodeError:
            print(response_text)
            raise ValueError("Failed to parse response as JSON")
        except Exception as e:
            print(response_text)
            raise ValueError(f"Error processing response: {str(e)}")

    def _prepare_image(self):
        self.img = self.image.copy()
        original_width, original_height = self.img.size
        new_height = int(self.sclaled_image_size * self.img.size[1] / self.img.size[0])
        
        self.scaled_img = self.img.resize((self.sclaled_image_size, new_height), Image.Resampling.LANCZOS)
    
        self.scale_x = original_width / self.sclaled_image_size
        self.scale_y = original_height / new_height
        self.original_width = original_width
        self.original_height = original_height

    def _get_bounding_box(self, bbox):
        boxlabel = "box_2d"
        
        if boxlabel not in bbox:
            raise ValueError(f"Bounding box doesn't contain {boxlabel}: {bbox}")
            
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int((bbox[boxlabel][0]/1000 * self.original_height))
        abs_x1 = int((bbox[boxlabel][1]/1000 * self.original_width)) 
        abs_y2 = int((bbox[boxlabel][2]/1000 * self.original_height))
        abs_x2 = int((bbox[boxlabel][3]/1000 * self.original_width))
        return {"y1": abs_y1, "x1": abs_x1, "y2": abs_y2, "x2": abs_x2}

    def detect_ui_elements(self) -> List[Dict]:
        if not self.override_cache:
            cached_results = self._get_cached_ui_elements()
            if cached_results:
                return cached_results

        response = self.model.generate_content(self.ui_detect_prompt, self.scaled_img)
        ui_elements = self._parse_response(response.text)

        print(f"Detected {len(ui_elements)} UI elements")
        self._cache_ui_elements(ui_elements)

        return ui_elements

    @staticmethod
    def parse_json(json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        try:
            json_output = json.loads(json_output)
        except json.JSONDecodeError:
            print(json_output)
            raise ValueError("Failed to parse JSON output")
        return json_output


    def detect_bounding_box(self, element: UIElement) -> bool:
        if not self.override_cache:
            cached_results = self._get_cached_bounding_box(element["element_id"])
            if cached_results:
                return cached_results
            
        prompt = f"""
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Only return the bounding box as 'box_2d'. 
        Focus a single requested UI element and its position. Make sure to cover the entire element in the bounding box.
        Detect the 2d bounding box for: {element["visual_description"]}"""

        response = self.model.generate_content(prompt, self.scaled_img)
        result = self.parse_json(response.text)
        if not result:
            raise ValueError(f"Failed to detect bounding box for {element}")
        bbox = result[0]

        print("result: ", bbox)
        return self._get_bounding_box(bbox)

    def get_click_coordinates(self, element: UIElement) -> Dict[str, int]:
        # Implementation for getting coordinates
        # Returns {"x": x_coord, "y": y_coord}
        return None

    def get_click_coordinates_by_description(self, description: str) -> Dict[str, int]:
        # Implementation for getting coordinates by description
        # Returns {"x": x_coord, "y": y_coord}
        return {"x": 0, "y": 0}  # Placeholder implementation

if __name__ == "__main__":

    # set -gx PYTHONPATH $PYTHONPATH /Users/klaus/Library/CloudStorage/OneDrive-Persönlich/Coding/Emulation/emulator-control-ng
    import json
    import os

    with open('config.json') as config_file:
        config = json.load(config_file)

    api_key = config.get("google_api_key")
    if not api_key:
        raise ValueError("API key not found in config.json.")

    vision = GoogleVision(api_key)

    screenshots = os.listdir("test_data")
    screenshots = [s for s in screenshots if "_marked" in s]
    for screenshot in screenshots:
        try:
            img = Image.open(f"test_data/{screenshot}")
            vision.add_screen(img)
            ui_elements = vision.detect_ui_elements()
            print(f"UI elements for {screenshot}:")
            for elem in ui_elements:
                vision.verify_element(elem)
        except Exception as e:
            print(f"Error processing {screenshot}: {str(e)}")

    