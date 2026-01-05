
import cv2
import numpy as np
import argparse
from pathlib import Path
from pdf_loader import load_pdf, render_page
from image_processor import preprocess_from_array, get_character_candidates, extract_candidate_roi
from template_manager import TemplateManager
from matcher import match_character

def debug_candidates(pdf_path, page_num, zoom=6.0, templates_dir="templates"):
    # 1. Load Templates
    tm = TemplateManager()
    tm.load_templates_from_dir(templates_dir)
    print(f"Loaded {len(tm.templates)} templates.")

    doc = load_pdf(pdf_path)
    if not doc:
        print("Failed to load PDF")
        return

    print(f"Rendering Page {page_num} at Zoom {zoom}...")
    img = render_page(doc, page_num - 1, zoom=zoom) 
    
    if img is None:
        print("Failed to render page.")
        return

    # 2. Preprocess (Same as main.py)
    _, gray, binary = preprocess_from_array(img)
    
    # Save binary for inspection
    cv2.imwrite("debug_binary.png", binary)
    
    # 3. Get Candidates (Same logic as main.py)
    candidates = get_character_candidates(binary)
    print(f"Total Candidates Found: {len(candidates)}")
    
    vis_img = img.copy()
    
    accepted = 0
    rejected = 0
    
    print("-" * 60)
    print(f"{'Char':<6} | {'Score':<6} | {'W':<5} | {'H':<5} | {'Status'}")
    print("-" * 60)
    
    for cand in candidates:
        x, y, w, h = cand['bbox']
        
        # 4. Match against Templates (New feature)
        roi = extract_candidate_roi(gray, cand)
        best_char, best_score = match_character(roi, tm.templates)
        
        is_match = False
        if best_score > 0.5:
            is_match = True
            accepted += 1
            # Green for Match
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{best_char} {best_score:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            status = "MATCH"
        else:
            rejected += 1
            # Red for No Match (but was a candidate)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            status = "NO_MATCH"
            
        print(f"{best_char if is_match else '?' :<6} | {best_score:.2f}   | {w:<5} | {h:<5} | {status}")

    print("-" * 60)
    print(f"Matched: {accepted}")
    print(f"Unmatched (Noise): {rejected}")
    
    cv2.imwrite("debug_contours.png", vis_img)
    print("Saved debug_contours.png and debug_binary.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--page", type=int, default=26)
    parser.add_argument("--zoom", type=float, default=6.0)
    parser.add_argument("--templates", default="templates")
    args = parser.parse_args()
    
    debug_candidates(args.pdf_path, args.page, args.zoom, args.templates)
