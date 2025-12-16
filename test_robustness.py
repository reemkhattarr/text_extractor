import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import string
import sys
sys.path.append(os.getcwd())
from template_manager import TemplateManager

def create_rotated_data(font_path, output_img="test_board_rotated.png"):
    # Generate templates if not exist (standard ones)
    tm = TemplateManager()
    chars = string.ascii_uppercase + string.digits
    if not os.path.exists("templates") or not os.listdir("templates"):
         tm.generate_templates_from_font(font_path, chars, size=32, output_dir="templates")
    
    # Create Board Image
    w, h = 800, 800
    img = Image.new('RGB', (w, h), (0, 100, 0)) # Green board
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 32)
    except Exception:
        print("Font error")
        return

    # Helper to draw rotated text
    def draw_rotated_text(image, text, position, angle, font, fill=(255,255,255)):
        # Create a separate image for the text
        # Get text size
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Make a mask image large enough
        mask = Image.new('L', (text_w + 10, text_h + 10), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.text((0, 0), text, font=font, fill=255)
        
        # Rotate
        rotated_mask = mask.rotate(angle, expand=True)
        
        # Paste onto main image
        dest_x, dest_y = position
        # We need to paste using the mask as the alpha
        color_img = Image.new('RGB', rotated_mask.size, fill)
        image.paste(color_img, (dest_x, dest_y), rotated_mask)

    # Draw text
    # Standard 0 deg
    draw.text((100, 100), "R1", font=font, fill=(255, 255, 255))
    
    # 90 deg (Vertical down)
    draw_rotated_text(img, "C2", (100, 300), 90, font)

    # 180 deg (Upside down)
    draw_rotated_text(img, "U3", (400, 400), 180, font)

    # 270 deg (Vertical up)
    draw_rotated_text(img, "L4", (600, 300), 270, font)
    
    # Mixed
    draw_rotated_text(img, "R90", (300, 600), 90, font)
        
    img.save(output_img)
    print(f"Created {output_img}")

if __name__ == "__main__":
    font = r"C:\Windows\Fonts\arial.ttf"
    if not os.path.exists(font):
        font = r"C:\Windows\Fonts\calibri.ttf"
    create_rotated_data(font)
