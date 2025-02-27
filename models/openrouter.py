from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
import base64
from typing import Optional, Union, List, Dict, Any
from models.api import Model, retry_with_backoff
from openai import OpenAI

from util.logger import general_logger as logger
from util.logger import llm_logger

class OpenRouter(Model):
    def __init__(self, api_key: str, model_name, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        
    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

    @retry_with_backoff(retries=5, initial_delay=2)
    def complete(self, prompt: str, system_prompt: Optional[str] = None, schema: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt}
            ],
        })
       
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            if not response or not response.choices:
                raise Exception("Empty response received from the model")
            
            llm_logger.debug(f"Calling OpenRouter {self.model_name} API with prompt:\n {prompt}\nResponse:\n {response}\n\n")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            raise

    @retry_with_backoff()
    def generate_content(self,
                        prompt: str,
                        images: Union[Image.Image, List[Image.Image]],
                        system_prompt: Optional[str] = None,
                        schema: Optional[str] = None) -> str:

        # Convert single image to list for consistent handling
        if not isinstance(images, list):
            images = [images]

        try:
            # Prepare content array starting with images
            content = []
            
            # Add each image to the content array
            for img in images:
                encoded_image = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                })
            
            # Add the text prompt after all images
            content.append({"type": "text", "text": prompt})
            
            # Prepare messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": content,
            })

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            if not response:
                raise Exception("Empty response received from the model")
            
            llm_logger.debug(f"Calling OpenRouter {self.model_name} API with prompt:\n {prompt}\nResponse:\n {response}\n\n")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in generate_content: {str(e)}")
            raise








