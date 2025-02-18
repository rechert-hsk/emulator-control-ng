from typing import List, Optional, Generator, Dict
from dataclasses import dataclass
import json
import time
from models.api import Model
from protocol.vnc_controller import VNCController
from task_execution.data_structures import StepPlan
from util.logger import event_logger


@dataclass
class VNCEvent:
    """Represents a primitive VNC operation"""
    type: str  # mousemove, mousedown, mouseup, keydown, keyup, delay
    x: Optional[int] = None
    y: Optional[int] = None
    key: Optional[str] = None
    duration: Optional[float] = None
    button: Optional[str] = None  # left, middle, right


class EventManager:
    """
    Manages the translation and execution of high-level UI interactions into
    primitive VNC events.
    """
    
    def __init__(self, model, vnc_controller: VNCController):
        self.vnc = vnc_controller
        self.default_delay = 0.1
        self.max_retries = 3
        
        self.translator_model = model

        self.system_prompt="""You are an expert in translating UI interactions 
            into primitive VNC events. Break down complex UI interactions into
            precise sequences of basic mouse and keyboard operations.
            
            Available primitive events:
            - mousemove(x, y)
            - mousedown(button)
            - mouseup(button)
            - keydown(key)
            - keyup(key)
            - delay(duration)
            
            Return sequences as JSON arrays of event objects.
            Consider proper timing, keyboard modifiers, and mouse positioning.
            """

    def execute_instructions(self, plan: StepPlan) -> bool:
        """Execute a sequence of instructions"""
        
        instructions = plan.steps
        for instruction in instructions:
            retry_count = 0
            success = False
            
            while not success and retry_count < self.max_retries:
                try:
                    context = {
                        "action": instruction.action,
                        "target": instruction.target,
                        "coordinates": instruction.coordinates,
                        "context": instruction.context or {},
                        "interaction_details": instruction.interaction_details or {}
                    }
                    
                    events = self._handle_dynamic_interaction(
                        description=str(instruction.target),
                        context=context
                    )
                    print(f"Executing instruction: {instruction}")
                    success = self._execute_event_sequence(events)
                        
                except Exception as e:
                    event_logger.error(f"Execution failed (attempt {retry_count + 1}): {str(e)}")
                    success = False
                
                if not success:
                    retry_count += 1
                    time.sleep(self.default_delay * 2)
            
            if not success:
                return False
                
        return True

    def _execute_event_sequence(self, events: Generator[VNCEvent, None, None]) -> bool:
        """Execute a sequence of VNC events"""
        try:
            for event in events:
                success = self._execute_vnc_event(event)
                if not success:
                    return False
                time.sleep(self.default_delay)
            return True
        except Exception as e:
            event_logger.error(f"Event sequence execution failed: {str(e)}")
            return False

    def _execute_vnc_event(self, event: VNCEvent) -> bool:
        """Execute a single VNC event"""
    
        try:
            if event.type == "mousemove":
                self.vnc.mouse_move(event.x, event.y)
            elif event.type == "mousedown":
                self.vnc.mouse_down(event.button)
            elif event.type == "mouseup":
                self.vnc.mouse_up(event.button)
            elif event.type == "keydown":
                self.vnc.key_down(event.key)
            elif event.type == "keyup":
                self.vnc.key_up(event.key)
            elif event.type == "delay":
                if event.duration and event.duration > 0:
                    time_to_sleep = min(event.duration/1000, 60)
                    time.sleep(time_to_sleep)
            return True
            
        except Exception as e:
            print(f"Failed to execute VNC event {event}: {str(e)}")
            return False

    def _handle_dynamic_interaction(self, description: str, context: Dict) -> Generator[VNCEvent, None, None]:
        """Generate VNC events for complex interactions using LLM"""
        prompt = f"""
Translate this UI interaction into a sequence of primitive VNC events.
INTERACTION: {description}
CONTEXT: {json.dumps(context, indent=2)}
REQUIREMENTS:
1. Only use these primitive event types:
- mousemove: Requires x,y coordinates
- mousedown: Requires button value (1=left, 2=middle, 3=right)
- mouseup: Requires button value (1=left, 2=middle, 3=right)
- keydown: Requires key value
- keyup: Requires key value
- delay: Requires duration in ms

2. For keyboard events, only use these special keys:
- Single characters (a-z, 0-9, punctuation)
- Special keys (must be exact match):
  * "return" or "enter" - For Enter/Return key
  * "tab" - For Tab key
  * "esc" - For Escape key
  * "bsp" - For Backspace
  * "delete" or "del" - For Delete
  * "home" - For Home key
  * "end" - For End key
  * "pgup" - For Page Up
  * "pgdn" - For Page Down
  * "left", "right", "up", "down" - For arrow keys
  * "space" or "spacebar" - For Space
  * "f1" through "f20" - For function keys
  * "shift", "ctrl", "alt" - For modifier keys
  * "lshift", "rshift", "lctrl", "rctrl", "lalt", "ralt" - For specific modifier sides

3. Include essential interaction components:
- Position mouse before any clicks
- Press/release modifier keys properly
- Add delays between actions (min 50ms)
- Handle multi-step sequences in correct order

4. IMPORTANT
- Carefully evaluate all information provided as context

RESPONSE FORMAT:
Return a JSON array of events where each event has:
{{
"type": "<event_type>",
"x": number, // For mouse events
"y": number, // For mouse events
"key": string, // For keyboard events (must match special keys list above)
"duration": number // For delays (in ms)
}}

Example:
[
{{"type": "keydown", "key": "shift"}},
{{"type": "keydown", "key": "a"}},
{{"type": "keyup", "key": "a"}},
{{"type": "keyup", "key": "shift"}},
{{"type": "delay", "duration": 50}},
{{"type": "mousemove", "x": 100, "y": 200}}
]
"""
        
        response = self.translator_model.complete(prompt, self.system_prompt)
        events_data = Model.parse_json(response)
        
        event_logger.debug(f"Dynamic interaction translation: {events_data}")

        # Buffer events to validate complete sequence before yielding
        validated_events = []
        for event_data in events_data:
            event = self._validate_and_create_event(event_data)
            if event is None:
                event_logger.error("Failed to validate complete event sequence")
                raise ValueError(f"Invalid event in sequence: {event_data}")
            validated_events.append(event)

            # Only yield events if entire sequence was validated
        for event in validated_events:
            event_logger.debug(f"Generated event: {event}")
            yield event
                

    def _validate_and_create_event(self, event_data: Dict) -> Optional[VNCEvent]:
        """Validate and create VNCEvent from dictionary"""
        required_fields = {
            "mousemove": ["x", "y"],
            "mousedown": ["button"],
            "mouseup": ["button"],
            "keydown": ["key"],
            "keyup": ["key"],
            "delay": ["duration"]
        }
        
        event_logger.debug(f"Validating event data: {event_data}")
        event_type = event_data.get("type")
        if event_type not in required_fields:
            event_logger.error(f"Invalid event type: {event_type}") 
            return None
            
        for field in required_fields[event_type]:
            if field not in event_data:
                event_logger.error(f"Missing required field '{field}' for event type '{event_type}'")
                return None
                
        return VNCEvent(**event_data)

    