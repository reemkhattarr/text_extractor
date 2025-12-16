import cv2
import cv2
import numpy as np

def remove_lines(binary, min_line_length=40, thickness=2):
    """
    Detects and removes long horizontal and vertical lines from the binary image.
    """
    # 1. Detect Horizontal Lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    
    # 2. Detect Vertical Lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
    # 3. Combine Lines
    lines = cv2.add(h_lines, v_lines)
    
    # 4. Dilate lines slightly to ensure clean removal
    kernel = np.ones((thickness, thickness), np.uint8)
    lines = cv2.dilate(lines, kernel, iterations=3)
    
    # 5. Subtract lines from original binary
    # We use bitwise_and with the INVERSE of the lines
    clean_binary = cv2.bitwise_and(binary, cv2.bitwise_not(lines))
    
    return clean_binary

def preprocess_from_array(img):
    """
    Preprocesses a loaded image (numpy array).
    """
    if img is None:
        raise ValueError("Image is None")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try OTSU thresholding
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we inverted it? (i.e. if background is white and text is black)
    # Heuristic: Count white pixels. If > 50%, invert.
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.size
    if white_pixels > total_pixels / 2:
        binary = cv2.bitwise_not(binary)
        
    # Remove large lines (circuits) to isolate text
    binary = remove_lines(binary, min_line_length=25)
        
    # Morphological operations to clean up
    # Closing to connect broken characters - BLOCKED: This merges text with lines in high-res diagrams
    # kernel = np.ones((3,3), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return img, gray, binary

def preprocess_image(image_path, debug=False):
    """
    Loads and preprocesses the PCB image.
    Args:
        image_path: Path to file.
    Returns: 
        img: Original color image
        gray: Grayscale image
        binary: Inverted thresholded image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    return preprocess_from_array(img)

def get_character_candidates(binary_image, min_w=5, min_h=8, max_w=200, max_h=200):
    """
    Finds countours in the binary image.
    Returns: list of dicts with bounding boxes and other info.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter noise by size
        if w >= min_w and h >= min_h and w <= max_w and h <= max_h:
            # We also might want to filter by aspect ratio if needed
            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'bbox': (x, y, w, h)
            })
            
    # Sort candidates (top to bottom, left to right) - helpful for debugging
    # But strictly, grouping logic will handle spatial relationships.
    return candidates

def extract_candidate_roi(gray_image, candidate, pad=2):
    """
    Extracts the ROI from the grayscale image for a given candidate.
    """
    x, y, w, h = candidate['bbox']
    
    # Add padding
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray_image.shape[1], x + w + pad)
    y2 = min(gray_image.shape[0], y + h + pad)
    
    return gray_image[y1:y2, x1:x2]
