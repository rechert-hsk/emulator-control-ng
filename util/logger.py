import logging
from typing import Optional
import os
import sys

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO, console_output: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Force recreation of handlers
    logger.handlers.clear()
    
    # Configure root logger if not already done
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.DEBUG)
    
    if log_file:
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            # Create file handler with absolute path
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Error setting up file handler for {name}: {str(e)}", file=sys.stderr)
    
    # Only add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
    
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging
    
    return logger


general_logger = get_logger('main', console_output=True)  # Only this one gets console output
llm_logger = get_logger('llm', log_file='logs/llm.log', level=logging.DEBUG)
event_logger = get_logger('events', log_file='logs/event.log', level=logging.DEBUG)
vision_logger = get_logger('vision', log_file='logs/vision.log', level=logging.DEBUG)
plan_logger = get_logger('plan', log_file='logs/plan.log', level=logging.DEBUG)

def test_logging():
    """Test function to verify logger configuration"""
    test_logger = get_logger('test', log_file='logs/test.log', level=logging.DEBUG)
    test_logger.debug('Debug message')
    test_logger.info('Info message')
    test_logger.warning('Warning message')
    test_logger.error('Error message')
    
    # Verify file contents
    if os.path.exists('logs/test.log'):
        with open('logs/test.log', 'r') as f:
            print("Log file contents:", file=sys.stderr)
            print(f.read(), file=sys.stderr)
    else:
        print("Log file not created!", file=sys.stderr)

if __name__ == "__main__":
    test_logging()