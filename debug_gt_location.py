import sys
from pathlib import Path
import fitz

def main():
    pdf_path = "pico-datasheet.pdf"
    page_num = 28 # 1-based
    
    # Target: R8 at (1134, 1393) in 6.0 zoom pixels
    # PDF coordinates (points)
    target_x = 1134 / 6.0
    target_y = 1393 / 6.0
    print(f"Looking for text near ({target_x:.2f}, {target_y:.2f}) on Page {page_num}")
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        
        words = page.get_text("words")
        # word: (x0, y0, x1, y1, text, ...)
        
        found = []
        for w in words:
            # Check if close (within 50 points)
            wx = (w[0] + w[2]) / 2
            wy = (w[1] + w[3]) / 2
            dist = ((wx - target_x)**2 + (wy - target_y)**2)**0.5
            
            if dist < 50:
                found.append((dist, w))
        
        found.sort(key=lambda x: x[0])
        
        print(f"Found {len(found)} words near target:")
        for dist, w in found:
            print(f"Dist: {dist:.2f} | Text: '{w[4]}' | Box: {w[:4]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
