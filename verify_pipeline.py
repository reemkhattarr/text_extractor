import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import string
# Add current dir to path to import template_manager
import sys
sys.path.append(os.getcwd())
from template_manager import TemplateManager

def create_synthetic_data(font_path, output_img="test_board.png"):
    # Generate templates
    print(f"Generating templates using {font_path}...")
    tm = TemplateManager()
    chars = string.ascii_uppercase + string.digits
    tm.generate_templates_from_font(font_path, chars, size=32, output_dir="templates")
    
    # Create Board Image
    w, h = 800, 600
    img = Image.new('RGB', (w, h), (0, 100, 0)) # Green board
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 32)
    except Exception as e:
        print(f"Font fail: {e}")
        return
        
    # Draw text
    labels = [("R101", 100, 100), ("C22", 300, 150), ("U1", 500, 300), ("L5", 100, 400)]
    
    for text, x, y in labels:
        # Draw white text (Silkscreen)
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
    img.save(output_img)
    print(f"Created {output_img}")

if __name__ == "__main__":
    font = r"C:\Windows\Fonts\arial.ttf"
    if not os.path.exists(font):
        # try another
        font = r"C:\Windows\Fonts\calibri.ttf"
        
    if not os.path.exists(font):
         print("Warning: Could not find Arial or Calibri. Please specify a font manually or ensure filtered directories.")
         # Fallback to simple generic search if needed or fail.
         
    create_synthetic_data(font)
