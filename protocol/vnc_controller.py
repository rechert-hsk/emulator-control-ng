from vncdotool import api, client
import os
from tempfile import gettempdir
from PIL import Image


class VNCController():
    def __init__(self, ) -> None:
        self.client = None
        self.framenumber = 0

    def connect(self, vnc_address="localhost::5900", password=None):
        self.client = api.connect(vnc_address, password)

    def get_screenshot(self) -> Image.Image:
        screenshot_file = self._allocate_screenshot_file()
        self.client.captureScreen(screenshot_file)
        self.framenumber += 1
        # Load and return PIL Image directly
        screenshot = Image.open(screenshot_file)
        # Clean up temporary file
        os.remove(screenshot_file)
        return screenshot

    def _allocate_screenshot_file(self):
            return os.path.join(gettempdir(), f"vnc_screenshot_{self.framenumber}.png")
            
    def mouse_move(self, target_x, target_y):
        self.client.mouseMove(int(target_x), int(target_y))

    def mouse_down(self, button):
        self.client.mouseDown(button)

    def mouse_up(self, button):
        self.client.mouseUp(button)

    def key_down(self, key):
        # Convert key to lowercase to match keymap
        key = str(key).lower()
        # Keys longer than 1 character must be in the keymap
        if len(key) > 1 and key not in client.KEYMAP:
            raise ValueError(f"Invalid key: {key}. Must be single character or one of {list(client.KEYMAP.keys())}")
        self.client.keyDown(key)
    
    def key_up(self, key):
        # Convert key to lowercase to match keymap
        key = str(key).lower()
        # Keys longer than 1 character must be in the keymap
        if len(key) > 1 and key not in client.KEYMAP:
            raise ValueError(f"Invalid key: {key}. Must be single character or one of {list(client.KEYMAP.keys())}")
        self.client.keyUp(key)