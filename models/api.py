from abc import ABC, abstractmethod
from PIL import Image
from typing import Optional, Union, List, Dict, Any
import json
import os

import time
import logging
from typing import Optional, Union, List
from PIL import Image

logger = logging.getLogger(__name__)

def retry_with_backoff(retries=5, initial_delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
                    
                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2  # exponential backoff
                        
            logger.error(f"All {retries} attempts failed. Last error: {str(last_exception)}")
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("All retry attempts failed without specific error")
            
        return wrapper
    return decorator

class Model(ABC):
    """Abstract base class for different model implementations"""
    
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
            return json_output
        except json.JSONDecodeError as e:
            print(json_output)
            raise ValueError("Failed to parse JSON output: " + str(e))

    @abstractmethod
    def complete(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 schema: Optional[str] = None) -> str:
        """Complete a prompt based on system and user messages"""
        pass

    @abstractmethod
    def generate_content(self, 
                        prompt: str, 
                        images: Union[Image.Image, List[Image.Image]], 
                        schema: Optional[str] = None, 
                        system_prompt: Optional[str] = None) -> str:
        """Generate content based on prompt and images"""
        pass
    
    @staticmethod
    def create_from_config(model_type: str) -> 'Model':
        """
        Factory method to create a model instance based on configuration
        
        Args:
            model_type: Type of model to create ('planning_model' or 'label_model')
            
        Returns:
            An instance of a Model implementation
        """
        # Get the absolute path to the config file
        config_path = os.path.join(os.getcwd(), "config.json")
        
        # Load the configuration
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Configuration file at {config_path} is not valid JSON")
        
        if model_type not in config:
            raise ValueError(f"Model type '{model_type}' not found in configuration")
        
        model_config = config[model_type]
        provider = model_config.get("provider", "").lower()
        
        # Import the existing OpenRouter implementation
        from models.openrouter import OpenRouter
        
        if provider == "openrouter":
            return OpenRouter(
                api_key=model_config["apiKey"],
                model_name=model_config["model"]
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Currently only 'openrouter' is supported.")