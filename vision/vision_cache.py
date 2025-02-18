from typing import Dict, List, Optional, Tuple
import os
import json
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import hashlib
from dataclasses import dataclass
from enum import Enum
import io

class CacheType(Enum):
    UI_ELEMENTS = "ui_elements"
    CLICK_COORDINATES = "click_coordinates"
    BOUNDING_BOX = "bounding_box"

@dataclass
class CacheEntry:
    image_hash: str
    ui_elements: Optional[List[Dict]] = None
    click_coordinates: Dict[str, Dict[str, int]] = None
    bounding_box: Dict[str, Dict[str, int]] = None
    
    def to_dict(self) -> Dict:
        return {
            "image_hash": self.image_hash,
            "ui_elements": self.ui_elements,
            "click_coordinates": self.click_coordinates,
            "bounding_box": self.bounding_box
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'CacheEntry':
        return CacheEntry(
            image_hash=data["image_hash"],
            ui_elements=data.get("ui_elements"),
            click_coordinates=data.get("click_coordinates", {}),
            bounding_box=data.get("bounding_box", {})
        )

class VisionCache:
    def __init__(self, cache_dir: str = ".vision_cache"):
        self.cache_dir = cache_dir
        self.image_dir = os.path.join(cache_dir, "images")
        self.data_dir = os.path.join(cache_dir, "data")
        self._init_cache_dirs()
        self.cached_images = {}  # Store loaded images for SSIM comparison
        self.cache_entries: Dict[str, CacheEntry] = {}
        self._load_cache()
        
    def _init_cache_dirs(self):
        """Initialize cache directory structure"""
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _load_cache(self):
        """Load existing cache entries"""
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    entry = CacheEntry.from_dict(data)
                    self.cache_entries[entry.image_hash] = entry
                    
                # Load image for SSIM comparison
                image_path = os.path.join(self.image_dir, f"{entry.image_hash}.png")
                if os.path.exists(image_path):
                    self.cached_images[image_path] = cv2.imread(image_path)
    
    def _save_cache_entry(self, entry: CacheEntry):
        """Save a cache entry to disk"""
        cache_path = os.path.join(self.data_dir, f"{entry.image_hash}.json")
        with open(cache_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
            
    def _compare_images(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.98) -> bool:
        """Compare two images using SSIM"""
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        score = ssim(img1, img2)
        # print(f"SSIM: {score}")
        return score >= threshold
        
    def find_similar_image(self, image: Image.Image) -> Tuple[str, CacheEntry]:
        """Find a similar image in cache using SSIM and return its hash and cache entry"""
        # Convert PIL Image to cv2 format
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for cached_path, cached_img in self.cached_images.items():
            if self._compare_images(cv2_image, cached_img):
                image_hash = os.path.basename(cached_path).split('.')[0]
                return image_hash, self.cache_entries.get(image_hash)
        return None, None
        
    def cache_result(self, image: Image.Image,
                    cache_type: CacheType,
                    data: Dict,
                    override: bool = False) -> str:
        """Cache results for an image
        
        Args:
            image: PIL Image object
            cache_type: Type of data being cached (UI_ELEMENTS or CLICK_COORDINATES)
            data: The data to cache
            override: If True, forces override of existing cached data
            
        Returns:
            str: Hash of the cached image
        """
        # Convert image to bytes for hashing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_hash = hashlib.md5(img_byte_arr.getvalue()).hexdigest()

        # Cache the image if not already cached or if override
        image_cache_path = os.path.join(self.image_dir, f"{image_hash}.png")
        if not os.path.exists(image_cache_path) or override:
            image.save(image_cache_path)
            # Convert PIL image to cv2 format for caching
            self.cached_images[image_cache_path] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Get or create cache entry
        entry = self.cache_entries.get(image_hash, CacheEntry(image_hash=image_hash))
        
        # Update appropriate data based on cache type
        if cache_type == CacheType.UI_ELEMENTS:
            if override or entry.ui_elements is None:
                entry.ui_elements = data
        elif cache_type == CacheType.CLICK_COORDINATES:
            if entry.click_coordinates is None:
                entry.click_coordinates = {}
            if override:
                # For override, completely replace the coordinates
                entry.click_coordinates = data
            else:
                # Without override, only add new coordinates
                entry.click_coordinates.update(data)
        elif cache_type == CacheType.BOUNDING_BOX:
            if entry.bounding_box is None:
                entry.bounding_box = {}
            if override:
                entry.bounding_box = data
            else:
                entry.bounding_box.update(data)
        
        # Save updated entry
        self.cache_entries[image_hash] = entry
        self._save_cache_entry(entry)
        
        return image_hash
        
    def get_cached_result(self, image: Image.Image, cache_type: CacheType) -> Dict:
        """Get cached results for an image if available"""
        image_hash, entry = self.find_similar_image(image)
        
        if entry:
            if cache_type == CacheType.UI_ELEMENTS:
                return entry.ui_elements
            elif cache_type == CacheType.CLICK_COORDINATES:
                return entry.click_coordinates
            elif cache_type == CacheType.BOUNDING_BOX:
                return entry.bounding_box
                
        return None