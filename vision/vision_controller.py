from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from models.api import Model
from vision.vision_base import UIElement, VisionLLMBase
from protocol.vnc_controller import VNCController
from skimage.metrics import structural_similarity as ssim
import cv2
import time
from PIL import Image
import numpy as np
from vision.vision_analysis import ScreenAnalysis
from util.logger import vision_logger

    
class VisionController:
    """Main controller class for managing vision-based UI analysis"""
    
    def __init__(self, vnc_controller:  VNCController, 
                model: Model,
                vision_provider_elements: VisionLLMBase, 
                vision_provider_coords: Optional[VisionLLMBase] = None,
                fps: float = 1.0, 
                stability_threshold: float = 0.98, 
                change_threshold: float = 0.98):
        
        self.vnc = vnc_controller
        self.frame_interval = 1.0 / fps
        self.stability_threshold = stability_threshold
        self.min_stable_frames = 3
        self.stable_frame_count = 0
        self.last_frame = None
        self.last_resolution: Optional[Tuple[int, int]] = None
        self.model = model
        self.change_threshold = change_threshold
        self.last_provided_frame = None
        self.new_stable_frame_available = False
        self.stop_processing = False

        self.provider = vision_provider_elements
        if vision_provider_coords is None:
            self.provider_coords = vision_provider_elements
        else:
            self.provider_coords = vision_provider_coords

    
    def calculate_frame_similarity(self, frame1: Image.Image, frame2: Image.Image) -> float:
        """Calculate similarity between two frames using SSIM"""
        # Convert PIL Images to numpy arrays in grayscale
        gray1 = np.array(frame1.convert('L'))
        gray2 = np.array(frame2.convert('L'))
        
        # Calculate SSIM with frames at their current resolution
        result = ssim(gray1, gray2, data_range=255, full=True)
        if isinstance(result, tuple):
            score = result[0]
        else:
            score = result
        return score
    
    def has_significant_change(self, current_frame: Image.Image) -> bool:
        """Determine if the current frame is significantly different from the last provided frame"""
        if self.last_frame is None:
            return True
            
        # First check if resolutions are different - automatic significant change
        if current_frame.size != self.last_frame.size:
            return True
            
        # Only do SSIM comparison if resolutions match
        similarity = self.calculate_frame_similarity(self.last_frame, current_frame)
        return similarity < self.change_threshold

    def wait_for_next_stable_frame(self, timeout: float = 30.0) -> Optional[ScreenAnalysis]:
        """
        Wait for the next meaningful stable frame (different from the last provided frame)
        Returns:
            - ScreenAnalysis object if a new stable frame is found
            - None if timeout is reached
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.new_stable_frame_available:
                self.new_stable_frame_available = False        
                return ScreenAnalysis(self.current_stable_frame, self.model, self.provider, self.provider_coords)
            time.sleep(0.2)
            
        return None

    def discard_stable_frame(self):
        """Discard the current stable frame"""
        self.stable_frame_count = 0
        self.new_stable_frame_available = False

    def process_continuous_screenshots(self):
        """Process screenshots continuously and detect stable frames with meaningful changes"""
        while not self.stop_processing:
            start_time = time.time()
            
            current_frame = self.vnc.get_screenshot()
            vision_logger.debug(f"Got screenshot at: {time.time()}")

            if self.has_significant_change(current_frame):
                self.stable_frame_count = 0
                self.new_stable_frame_available = False
            else:
                self.stable_frame_count += 1
                if self.stable_frame_count >= self.min_stable_frames:
                    self.current_stable_frame = current_frame.copy()
                    self.new_stable_frame_available = True
                    self.stable_frame_count = 0
                    vision_logger.debug("New stable frame ready after change")
            
            self.last_frame = current_frame.copy()
                
            # Maintain desired FPS
            elapsed_time = time.time() - start_time
            if elapsed_time < self.frame_interval:
                time.sleep(self.frame_interval - elapsed_time)