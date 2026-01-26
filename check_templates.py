import cv2
import os
import sys

def check_templates(dir_path):
    with open("template_sizes.txt", "w") as f:
        f.write(f"Checking templates in {dir_path}\n")
        files = os.listdir(dir_path)
        min_h = 1000
        min_w = 1000
        max_h = 0
        max_w = 0
        
        vals = []
        
        for f in files:
            if not f.endswith(".png"): continue
            path = os.path.join(dir_path, f)
            img = cv2.imread(path)
            if img is None: continue
            h, w = img.shape[:2]
            f.write(f"Template {f}: {w}x{h}\n")
            vals.append((w, h))
            min_h = min(min_h, h)
            min_w = min(min_w, w)
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        f.write(f"Min W: {min_w}, Min H: {min_h}\n")
        f.write(f"Max W: {max_w}, Max H: {max_h}\n")

        # Check 1
        t1 = os.path.join(dir_path, "1.png")
        if os.path.exists(t1):
            img = cv2.imread(t1)
            h, w = img.shape[:2]
            f.write(f"'1' Dimensions: {w}x{h}, Ratio: {w/h:.2f}\n")

if __name__ == "__main__":
    check_templates("schematics_templates")
