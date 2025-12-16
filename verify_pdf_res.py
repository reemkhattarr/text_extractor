import fitz
import cv2
import sys
import os
from pdf_loader import load_pdf, render_clip, render_page

def verify():
    path = "pico-datasheet.pdf"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    doc = load_pdf(path)
    if not doc:
        print("Failed to load PDF")
        return

    # Define a rect in points (x0, y0, x1, y1)
    # Let's take a 50x50 point area (e.g. some text)
    rect = (50, 50, 150, 150)
    
    # 1. High Res Clip (Zoom 24.0)
    # Expected size approx 100*24 = 2400 pix
    print("Rendering high res clip...")
    try:
        hi_res = render_clip(doc, 0, rect, zoom=24.0)
        print(f"High Res shape: {hi_res.shape}")
        cv2.imwrite("verify_hi_res.png", hi_res)
    except Exception as e:
        print(f"Error in render_clip: {e}")
        return
    
    # 2. Low Res Page (Zoom 2.0)
    # Expected size approx 100*2 = 200 pix for that region
    print("Rendering low res page...")
    try:
        full_page = render_page(doc, 0, zoom=2.0)
        # Crop it. rect is signals. 
        # x0_img = 50 * 2 = 100.
        x0 = int(rect[0] * 2.0)
        y0 = int(rect[1] * 2.0)
        x1 = int(rect[2] * 2.0)
        y1 = int(rect[3] * 2.0)
        
        # Clip to image bounds just in case
        h, w = full_page.shape[:2]
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))

        lo_res_crop = full_page[y0:y1, x0:x1]
        print(f"Low Res Crop shape: {lo_res_crop.shape}")
        cv2.imwrite("verify_lo_res.png", lo_res_crop)
    except Exception as e:
        print(f"Error in render_page/cropping: {e}")
        return
    
    if hi_res.shape[0] > lo_res_crop.shape[0] * 10:
        print("SUCCESS: High res image is massively larger (approx 12x).")
    else:
        print(f"FAILURE: High res image is not larger as expected. Hi: {hi_res.shape}, Lo: {lo_res_crop.shape}")

if __name__ == "__main__":
    verify()
