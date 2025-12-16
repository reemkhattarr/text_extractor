
import cv2
import numpy as np
import argparse
from pdf_loader import load_pdf, render_page
from image_processor import preprocess_from_array

def debug_candidates(pdf_path, page_num, zoom=6.0):
    doc = load_pdf(pdf_path)
    if not doc:
        print("Failed to load PDF")
        return

    print(f"Rendering Page {page_num} at Zoom {zoom}...")
    img = render_page(doc, page_num - 1, zoom=zoom) # page_num is 1-based in args usually, 0-based in fitz
    
    if img is None:
        print("Failed to render page.")
        return

    _, _, binary = preprocess_from_array(img)
    
    # Save binary for inspection
    cv2.imwrite("debug_binary.png", binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total Contours Found: {len(contours)}")
    
    accepted = 0
    rejected = 0
    
    # Defaults from image_processor.py
    min_w=5
    min_h=8
    max_w=200
    max_h=200
    
    vis_img = img.copy()
    
    print("-" * 40)
    print(f"{'Status':<10} | {'W':<5} | {'H':<5} | {'Aspect':<6} | {'Reason'}")
    print("-" * 40)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        reason = ""
        is_accepted = True
        
        if w < min_w: 
            reason += "TooSmallW "
            is_accepted = False
        if h < min_h:
            reason += "TooSmallH "
            is_accepted = False
        if w > max_w:
            reason += "TooBigW "
            is_accepted = False
        if h > max_h:
            reason += "TooBigH "
            is_accepted = False
            
        if is_accepted:
            accepted += 1
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green for Accepted
        else:
            rejected += 1
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red for Rejected
            
            # Label the reason if it's too big (interesting ones)
            if "TooBig" in reason:
                cv2.putText(vis_img, reason, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print("-" * 40)
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    
    cv2.imwrite("debug_contours.png", vis_img)
    print("Saved debug_contours.png and debug_binary.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--page", type=int, default=26)
    parser.add_argument("--zoom", type=float, default=6.0)
    args = parser.parse_args()
    
    debug_candidates(args.pdf_path, args.page, args.zoom)
