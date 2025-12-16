import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

class TemplateManager:
    def __init__(self):
        self.templates = {}

    def load_templates_from_dir(self, dir_path):
        """Loads template images from a directory."""
        path = Path(dir_path)
        if not path.exists():
            print(f"Template directory {dir_path} not found.")
            return

        for img_file in path.glob("*.png"):
            # Filename should be the character, e.g., "R.png", "1.png"
            # Special handling might be needed for special chars if filenames are restrictive
            char = img_file.stem
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates[char] = img
        print(f"Loaded {len(self.templates)} templates.")

    def generate_templates_from_font(self, font_path, characters, size=32, output_dir="templates"):
        """Generates template images using a specific font."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            font = ImageFont.truetype(font_path, size)
        except IOError:
            print(f"Could not load font: {font_path}")
            return

        for char in characters:
            # Create a provisional image to determine valid size
            dummy_img = Image.new('L', (1, 1), 255)
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), char, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Create actual image
            # Add some padding
            img_w, img_h = w + 4, h + 4
            image = Image.new('L', (img_w, img_h), 0) # Black background
            draw = ImageDraw.Draw(image)
            
            # Draw white text
            draw.text((2, 0), char, font=font, fill=255) # White text
            
            # Invert for template matching (optional, depending on strategy)
            # Usually we match features or use raw intensity. 
            # If using binary image for matching:
            #   Text is black (0), Background is white (255) -> Invert -> Text 255, BG 0
            
            # Convert to numpy for OpenCV (if needed later) or save
            save_path = os.path.join(output_dir, f"{char}.png")
            image.save(save_path)
            
            
            # Load back into memory as cv2 image
            self.templates[char] = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
            
        print(f"Generated {len(characters)} templates in {output_dir}")

if __name__ == "__main__":
    import argparse
    import string
    
    parser = argparse.ArgumentParser(description="Generate OCR templates.")
    parser.add_argument("--font", required=True, help="Path to .ttf font file")
    parser.add_argument("--out", default="templates", help="Output directory")
    parser.add_argument("--size", type=int, default=32, help="Font size")
    args = parser.parse_args()
    
    manager = TemplateManager()
    chars = string.ascii_uppercase + string.digits + ".-"
    manager.generate_templates_from_font(args.font, chars, args.size, args.out)

