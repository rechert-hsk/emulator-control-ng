from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
import base64
import json
from typing import Optional, Union, List, Dict, Any
from models.api import Model
import google.generativeai as genai

class GoogleModel(Model):
    def __init__(self, apikey, model_name: str = "gemini-2.0-flash", tools: List = [], system_prompt: str = "You are a helpful assistant."):
        genai.configure(api_key=apikey)
        self.model = genai.GenerativeModel(model_name, tools=tools, system_instruction=system_prompt)
        
    def complete(self, prompt, system_prompt = None, schema = None):
        return None

    def generate_content(self, 
                        prompt: str, 
                        images: Union[Image.Image, List[Image.Image]], 
                        schema: Optional[str] = None,
                        system_prompt: Optional[str] = None) -> str:
        
        if not isinstance(images, list):
            images = [images]
            
        contents = [prompt, *images]
            
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 1.0,
        }
        if schema:
            generation_config["response_schema"] = schema
            
        error_count = 0
        while error_count < 10:
            try:
                response = self.model.generate_content(
                    contents, 
                    generation_config=genai.GenerationConfig(**generation_config)
                )
                return response
            except Exception as e:
                error_count += 1
                print(f"Error processing frame: {e}")
                import time
                time.sleep(5)
        raise ValueError("Failed to process frame after 10 attempts")

