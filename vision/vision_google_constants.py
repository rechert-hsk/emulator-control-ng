
BOUNDING_BOX_SYSTEM_INSTRUCTIONS = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Focus on UI elements and their positions. 
Make sure to cover the entire element in the bounding box.
"""

DEFAULT_SYSTEM_INSTRUCTION = "You are an operator of a computer system. You can control the mouse and keyboard. You can use the mouse and the keyboard. "

UI_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string"
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                        "enum": ["mouse", "keyboard"]
                    },
                    "element": {
                        "type": "string",
                        "description": "Description of the UI element being interacted with"
                    },
                    "value": {
                        "type": "string",
                        "description": "Text to type for type events",
                        "nullable": True
                    }
                },
                "required": ["event", "element"]
            }
        }
    },
    "required": ["description", "actions"]
}

UI_DETECT_PROMPT = """
System: You are a computer vision assistant specialized in analyzing user interfaces. You must respond only in valid JSON format, containing a list of actionable UI elements.

User: Analyze the following screenshot and create a list of all clickable UI elements. For each element, provide:
- element_id (unique)
- element_type (button, link, input, etc.)
- element_text (if visible)
- context (e.g., "in modal window", "in navigation bar")
- visual_description (brief description of appearance)
- importance (high, medium, low)

Return the result as a JSON array with these properties. Only include actionable elements that a user could interact with.

Expected response format:
{
    "ui_elements": [
    {
        "element_id": "unique_id",
        "element_type": "element_type",
        "element_text": "visible_text",
        "context": "location_context",
        "visual_description": "brief_description",
        "importance": "priority_level"
    }
    ]
} """

def create_ui_planning_prompt(task):
    prompt = f"""
    The task is: {task}. Analyze the screen of this computer system and provide the necessary actions to {task}.

    For each screen:
    1. Provide a brief description of what needs to be done on the current screen
    2. Determine if the interface is mouse-enabled or keyboard-only (e.g. DOS/terminal screens)
    3. List the sequence of actions needed, where each action should be either:
    - keyboard key press: For pressing specific keys like [enter] or [f8]:
      - Set element to null
      - Specify key(s) to press in value field (e.g. value: "[enter]" or "[alt+f4]") 
    - keyboard text input: For typing text into fields:
      - Specify target input field in element field (e.g. element: "Username field")
      - Put the text to type in value field (e.g. value: "John Smith")
    - mouse event (only if interface supports mouse control):
      - Specify UI element to click in the "element" field (e.g. element: "Next button")
    - no operation needed (noop):
      - When confirming no action is needed on the current screen
      - Set both element and value to null
      - Use this when double-checking reveals no action is required

    Guidelines:
    - First determine if mouse control is available:
      - For DOS/terminal screens or text-only interfaces: use only keyboard events
      - For graphical interfaces: use mix of mouse and keyboard as appropriate
    - Always verify if an action is actually needed:
      - If no action is required, emit a "noop" event
      - Only emit actual actions when they are necessary to progress
    - For keyboard key presses:
      - Always set element to null
      - Use lowercase for key names in square brackets in value field
      - Common keys: [enter], [tab], [space], [f1]-[f12], [alt+f4], etc.
    - For text input:
      - In mouse-enabled interfaces: always click field with mouse event first
      - In keyboard-only interfaces: use [tab]/arrows to navigate between fields
      - Always include element name and value to type
      - In a moouse enabled interface, if the field is prefilled, clear it first
    - For mouse events (only in supported interfaces):
      - Always specify the UI element in element field
      - Always add the type of the UI element, e.g. User Name text field, Submit button
      - Set value to null
    - Describe UI elements precisely and unambiguously
    - List actions in the exact sequence needed
    - Include all necessary actions to complete the screen
    """
    return prompt