import fitz  # PyMuPDF
import cv2
import numpy as np

def load_pdf(path):
    """
    Opens a PDF file and returns the document object.
    """
    try:
        doc = fitz.open(path)
        return doc
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return None

def render_page(doc, page_num, zoom=2.0):
    """
    Renders a specific page of the PDF to an OpenCV BGR image.
    Args:
        doc: fitz.Document object
        page_num: int, 0-indexed page number
        zoom: float, scale factor (2.0 = 144 DPI usually)
    Returns:
        numpy array (height, width, 3) representing BGR image
    """
    if page_num < 0 or page_num >= len(doc):
        return None
        
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    return _pixmap_to_bgr(pix)

def render_clip(doc, page_num, rect_points, zoom=8.0):
    """
    Renders a specific region of a page at high resolution.
    Args:
        doc: fitz.Document
        page_num: int
        rect_points: tuple (x0, y0, x1, y1) in unscaled PDF points
        zoom: float, scale factor
    Returns:
        numpy array (height, width, 3) representing BGR image
    """
    if page_num < 0 or page_num >= len(doc):
        return None
        
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    clip = fitz.Rect(rect_points)
    
    # get_pixmap with clip automatically handles the cropping and scaling
    pix = page.get_pixmap(matrix=mat, clip=clip)
    
    return _pixmap_to_bgr(pix)

def _pixmap_to_bgr(pix):
    """Helper to convert fitz pixmap to CV2 BGR image."""
    # Convert from RGB (PyMuPDF default) to BGR (OpenCV)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    
    # Reshape
    if pix.n == 4: # RGBA
        img_data = img_data.reshape((pix.h, pix.w, 4))
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: # RGB
        img_data = img_data.reshape((pix.h, pix.w, 3))
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        # Gray or other
        img_data = img_data.reshape((pix.h, pix.w, pix.n))
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        
    return img_bgr
