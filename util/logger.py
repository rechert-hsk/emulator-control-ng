import logging
from typing import Optional
import os

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.StreamHandler() if log_file is None else logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


general_logger = get_logger('main')
llm_logger = get_logger('llm', log_file='logs/llm.log', level=logging.DEBUG)
event_logger = get_logger('events', log_file='logs/event.log', level=logging.DEBUG)
vision_logger = get_logger('vision', log_file='logs/vision.log', level=logging.DEBUG)
plan_logger = get_logger('plan', log_file='logs/plan.log', level=logging.DEBUG)