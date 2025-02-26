from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass, fields
import json
from pathlib import Path
from vision.vision_cache import VisionCache, CacheType
import os
from PIL import Image
import torch
from lib.OmniParser.omni_utils import *
from PIL import Image


@dataclass
class UIElement:
    """Represents a UI element with its properties"""
    id: str
    type: str  # button, link, etc.
    text: Optional[str]  # visible text
    context: str  # e.g., "in modal", "in navbar"
    bounding_box: Optional[Dict[str, int]] = None
    visual_description: Optional[str] = None
    click_coordinates: Optional[Dict[str, int]] = None

    @staticmethod
    def normalize_element_data(elem_dict):
        normalized = {}
        for key, value in elem_dict.items():
            # Remove 'element_' prefix if it exists
            normalized_key = key.replace('element_', '')
            normalized[normalized_key] = value
        return normalized
    
    @staticmethod
    def from_dict(data: Dict) -> 'UIElement':
        normalized_data = UIElement.normalize_element_data(data)
        valid_keys = {field.name for field in fields(UIElement)}
        filtered_data = {k: v for k, v in normalized_data.items() if k in valid_keys}
        return UIElement(**filtered_data)

    def __str__(self):
        click_coords = self.click_coordinates if self.click_coordinates else "None"
        return (f"UIElement(id='{self.id}', type='{self.type}', text='{self.text}', "
                f"context='{self.context}', click_coordinates={click_coords})")


class VisionLLMBase(ABC):
    """Base class for LLM vision providers"""

    BOX_TRESHOLD=0.01
    model_path = "lib/OmniParser/weights/icon_detect/model.pt"
    som_model = get_yolo_model(model_path)

    def __init__(self):
        self.cache = VisionCache()
        self.image = None
        self.override_cache = False
        
    @abstractmethod
    def add_screen(self, image: Image.Image, override_cache: bool = False):
        self.image = image.convert("RGB") 
        self.override_cache = override_cache

    @abstractmethod
    def detect_ui_elements(self) -> List[Dict]:
        """Detect UI elements in the image"""
        pass

    @abstractmethod
    def get_click_coordinates_by_description(self, description: str) -> Dict[str, int]:
        """Get click coordinates by description"""
        pass

    def get_click_coordinates(self, element: UIElement) -> Dict[str, int]:
        """Get click coordinates for a specific element"""
        if not element.bounding_box:
            raise ValueError("Bounding box not found in element")
        
        bbox = element.bounding_box

        abs_x1 = bbox["x1"]
        abs_y1 = bbox["y1"]
        abs_x2 = bbox["x2"]
        abs_y2 = bbox["y2"]                      

        center_x = (abs_x1 + abs_x2) // 2
        center_y = (abs_y1 + abs_y2) // 2
    
        coords =  {"x": center_x, "y" : center_y}
        self._cache_coordinates(element.id, coords, self.override_cache)
        return coords

    def _get_cached_ui_elements(self) -> Optional[List[Dict]]:
        """Get cached UI elements"""
        return self.cache.get_cached_result(self.image, CacheType.UI_ELEMENTS)
        
    def _get_cached_coordinates(self, element_id: str) -> Optional[Dict[str, int]]:
        """Get cached click coordinates"""
        coords = self.cache.get_cached_result(self.image, CacheType.CLICK_COORDINATES)
        return coords.get(element_id) if coords else None
    
    def _get_cached_bounding_box(self, element_id) -> Optional[Dict[str, int]]:
        """Get cached bounding box"""
        data = self.cache.get_cached_result(self.image, CacheType.BOUNDING_BOX)
        return data.get(element_id) if data else None
        
    def _cache_ui_elements(self, results: List[Dict], override: bool = False):
        """Cache UI detection results"""
        print
        self.cache.cache_result(
            self.image,
            CacheType.UI_ELEMENTS,
            results,
            override
        )
        
    def _cache_coordinates(self, element_id: str, coordinates: Dict[str, int], override: bool = False):
        """Cache click coordinates"""
        self.cache.cache_result(
            self.image,
            CacheType.CLICK_COORDINATES,
            {element_id: coordinates},
            override
        )

    def _cache_boundingbox(self, element_id: str, bbox: Dict[str, int], override: bool = False):
        """Cache click coordinates"""
        self.cache.cache_result(
            self.image,
            CacheType.BOUNDING_BOX,
            {element_id: bbox},
            override
        )

    def get_confidence(self, element: UIElement) -> float:
        """Get confidence for a specific element"""
        pass