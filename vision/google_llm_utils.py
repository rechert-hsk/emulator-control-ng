import google.generativeai as genai
from PIL import Image
from io import BytesIO
import json
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor


# def process_(_model: any, _img: Image, _prompt: str) -> str:
#     error_count = 0
#     while error_count < 10:
#       try:
#         response = _model.generate_content([_prompt, _img], generation_config=genai.GenerationConfig(
#                         response_mime_type="application/json",
#             ),  )
#         return response
#       except Exception as e:
#         error_count += 1
#         print(f"Error processing frame: {e}")
#         time.sleep(5)

# def process_frame(_model: any, _img: Image, _prompt: str, _schema:str=ACTIONS_SCHEMA) -> str:
#     error_count = 0
#     while error_count < 10:
#       try:
#         response = _model.generate_content([_prompt, _img], generation_config=genai.GenerationConfig(
#                         response_mime_type="application/json", response_schema=_schema,
#             ),  )
#         return response
#       except Exception as e:
#         error_count += 1
#         print(f"Error processing frame: {e}")
#         time.sleep(5)

# def process_outcome(_model: any, _img: Image, _img2: Image,_prompt: str, actions: str) -> str:
#     error_count = 0
#     while error_count < 10:
#       try:
#         response = _model.generate_content([_prompt, _img, json.dumps(actions), _img2], generation_config=genai.GenerationConfig(
#                         response_mime_type="application/json",
#             ),  )
#         return response
#       except Exception as e:
#         error_count += 1
#         print(f"Error processing frame: {e}")
#         time.sleep(5)

# def process_frame_ui_analysis(_img, _prompt):
#     error_count = 0
#     while error_count < 10:
#       try:
#         _model = get_model(system_prompt=bounding_box_system_instructions)
#         response = _model.generate_content([_prompt, _img], generation_config=genai.GenerationConfig(
#                         temperature=0.5),)
#         return _img, response
#       except Exception as e:
#         error_count += 1
#         print(f"Error processing frame: {e}")
#         time.sleep(5)

def plot_bounding_boxes(im, bounding_boxes, boxlabel = "box_2d", scaleX = 1.0, scaleY = 1.0):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("Keyboard.ttf", size=14)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int((bounding_box[boxlabel][1]/1000 * height))
      abs_x1 = int((bounding_box[boxlabel][0]/1000 * width))
      abs_y2 = int((bounding_box[boxlabel][3]/1000 * height))
      abs_x2 = int((bounding_box[boxlabel][2]/1000 * width))

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)


additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def vision_target(im, bounding_boxes, boxlabel = "box_2d", scaleX=1.0, scaleY=1.0):
    """
    Returns the coordinates of the center of the first bounding box in an image.

    Args:
        im: PIL Image object
        bounding_boxes: A list of bounding boxes containing object positions in normalized [y1 x1 y2 x2] format.
        scaleX: Scale factor for x coordinates (default 1.0)
        scaleY: Scale factor for y coordinates (default 1.0)

    Returns:
        tuple: (center_x, center_y) coordinates of the first bounding box
    """
    width, height = im.size
    
    # Parse the bounding boxes JSON
    bounding_boxes = parse_json(bounding_boxes)
    boxes = json.loads(bounding_boxes)
    
    if not boxes:
        return None
        
    # Get first bounding box
    box = boxes[0]
    
    # Convert normalized coordinates to absolute coordinates
    abs_y1 = int((box[boxlabel][0]/1000 * height) * scaleY)
    abs_x1 = int((box[boxlabel][1]/1000 * width) * scaleX) 
    abs_y2 = int((box[boxlabel][2]/1000 * height) * scaleY)
    abs_x2 = int((box[boxlabel][3]/1000 * width) * scaleX)

    # Handle inverted coordinates
    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

    # Calculate center coordinates
    center_x = (abs_x1 + abs_x2) // 2
    center_y = (abs_y1 + abs_y2) // 2
    
    return (center_x, center_y)

