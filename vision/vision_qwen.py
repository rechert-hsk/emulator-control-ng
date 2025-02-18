from vision.vision_base import VisionLLMBase, UIElement
from typing import List, Dict
from PIL import Image
from io import BytesIO
import json
from models.openrouter import OpenRouter
import os
import time

class QwenVision(VisionLLMBase):
    def __init__(self, apikey: str):
        super().__init__()
        # self.model = OpenRouter(apikey, model_name="qwen/qwen2.5-vl-72b-instruct:free")
        self.model = OpenRouter(apikey, model_name="qwen2.5-vl-72b-instruct", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.override_cache = True
        
    def add_screen(self, image_path: str, override_cache: bool = False):
        super().add_screen(image_path, override_cache)
        self._prepare_image()

    def _prepare_image(self):
        self.img = self.image.copy()
        original_width, original_height = self.img.size
        
        self.original_width = original_width
        self.original_height = original_height

    def _get_bounding_box(self, bbox):
        boxlabel = "bbox_2d"
        
        if boxlabel not in bbox:
            raise ValueError(f"Bounding box doesn't contain {boxlabel}: {bbox}")
            
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = bbox[boxlabel][1] # int((bbox[boxlabel][1]/1000 * self.original_height))
        abs_x1 = bbox[boxlabel][0] # int((bbox[boxlabel][0]/1000 * self.original_width)) 
        abs_y2 = bbox[boxlabel][3] # int((bbox[boxlabel][3]/1000 * self.original_height))
        abs_x2 = bbox[boxlabel][2 ]# int((bbox[boxlabel][2]/1000 * self.original_width))
        return {"y1": abs_y1, "x1": abs_x1, "y2": abs_y2, "x2": abs_x2}

    def detect_ui_elements(self) -> List[Dict]:
        return None

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
        print(f"Detecting bounding box for {element}")
        if not self.override_cache:
            cached_results = self._get_cached_bounding_box(element.id)
            if cached_results:
                return cached_results
            
        prompt = f"""
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Only return the bounding box as 'bbox_2d'. 
        Focus a single requested UI element and its position. Make sure to cover the entire element in the bounding box.
        Detect the 2d bounding box for: {element.visual_description}"""

        attempts = 0
        while attempts < 10:
            try:
                response = self.model.generate_content(prompt, self.img)
                result = self.parse_json(response)
                if not result or len(result) == 0:
                    raise ValueError(f"Failed to detect bounding box for {element}")
                bbox = result[0]

                bounding_box = self._get_bounding_box(bbox)
                self._cache_boundingbox(element.id, bounding_box)
                return bounding_box
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")
                time.sleep(1)
        raise ValueError(f"Failed to detect bounding box for {element} after 10 attempts")

    def get_click_coordinates(self, element: UIElement) -> Dict[str, int]:
        print(f"Getting click coordinates for {element}")
        if not self.override_cache:
            cached_results = self._get_cached_coordinates(element.id)
            if cached_results:
                return cached_results
        
        if element.bounding_box == None:
            element.bounding_box = self.detect_bounding_box(element)
        return super().get_click_coordinates(element)

        

    