
from vision.vision_base import VisionLLMBase, UIElement
from models.uitars import UITarsModel
import os
from typing import List, Dict
from io import BytesIO
from PIL import Image
import math

class UITarsVision(VisionLLMBase):
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.model = UITarsModel(base_url="http://localhost:8000/v1", api_key="empty")
        self.img = None
        self.verbose = verbose

    def add_screen(self, image_path: str, override_cache: bool = False):
        super().add_screen(image_path, override_cache)
        self._prepare_image()

    def _prepare_image(self):
        self.img = self.image.copy()
        self.image_width, self.image_height = self.img.size

    @staticmethod
    def _parse_coordinates(result_string):
    # Remove parentheses and split by comma
        coords = result_string.strip("()").split(",")
        x = int(coords[0])
        y = int(coords[1])
        return (x, y)

    # Function to convert relative coordinates to absolute coordinates
    def _relative_to_absolute(self, relative_coords):
        x_relative, y_relative = relative_coords
        x_absolute = math.floor(self.image_width * x_relative / 1000)
        y_absolute = math.floor(self.image_height * y_relative / 1000)
        return (x_absolute, y_absolute)

    def get_click_coordinates(self, element: UIElement) -> Dict[str, int]:
        if not self.override_cache:
            cached_coords = self._get_cached_coordinates(element['element_id'])
            print(cached_coords)
            if cached_coords:
                return cached_coords
            
        prompt = f"Output only the coordinate of one point in your response. What element matches the following description: {element}?"
        response = self.model.generate_content(prompt, self.img)
        if self.verbose:
            print(response)

        # retrun a dictionary with the coordinates {'x': 100, 'y': 200}
        coordinates  = self._relative_to_absolute(self._parse_coordinates(response))
        dict_coords = {'x': coordinates[0], 'y': coordinates[1]}

        self._cache_coordinates(element['element_id'], dict_coords)
        return dict_coords
    
    def detect_ui_elements(self) -> List[Dict]:
        return None
