from PIL import Image, ImageDraw


def mark_and_display_image(image_path, absolute_coords):
    # Open the image using PIL
    image = Image.open(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    # Draw a red circle at the absolute coordinates
    draw = ImageDraw.Draw(image)
    draw.ellipse((absolute_coords[0] - 5, absolute_coords[1] - 5, absolute_coords[0] + 5, absolute_coords[1] + 5), fill="red", outline="red")
    
    # Save the image with the marked point
    marked_image_path = image_path.replace(".png", "_marked.png")
    image.save(marked_image_path)