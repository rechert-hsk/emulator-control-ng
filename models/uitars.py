from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
import base64
import json
from typing import Optional, Union, List, Dict, Any
from models.api import Model
import os

class UITarsModel(Model):
    def __init__(self, base_url: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.output_dir = Path("ui_eval")
        
    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')
        
    def generate_content(self, 
                        prompt: str, 
                        images: Union[Image.Image, List[Image.Image]], 
                        schema: Optional[str] = None) -> str:
        
        if isinstance(images, list):
            images = images[0]  # UITars only handles single image
            
        encoded_image = self._encode_image(images)
        
        response = self.client.chat.completions.create(
            model="ui-tars",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                        {"type": "text", "text": prompt}
                    ],
                },
            ],
            frequency_penalty=1,
            max_tokens=128,
        )
        
        return response.choices[0].message.content