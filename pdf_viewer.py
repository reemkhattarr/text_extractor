import cv2
import numpy as np
from pdf_loader import render_page

def select_page_from_pdf(doc, start_page=0):
    """
    Opens a window to allow the user to navigate through PDF pages.
    Returns the index of the selected page (0-based) or None if cancelled.
    
    Controls:
    - 'n' or Right Arrow: Next Page
    - 'b' or Left Arrow: Previous Page
    - Mouse Wheel: Scroll Pages
    - Enter: Confirm Selection
    - Esc: Cancel
    """
    current_page = start_page
    total_pages = len(doc)
    window_name = "PDF Page Selector"
    
    # Callback state
    params = {'scroll_delta': 0}
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 is scroll up, flags < 0 is scroll down
            if flags > 0: param['scroll_delta'] = 1
            else: param['scroll_delta'] = -1

    # WINDOW_AUTOSIZE ensures 1:1 pixel mapping (no stretching/distortion)
    # This keeps "dimensions normal" logic requested by user
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback, params)
    
    # Zoom 1.3 provides a good balance for readability on 1080p screens without being too huge
    # A4 @ 1.3 is roughly 770x1100 pixels
    view_zoom = 1.3
    
    print(f"Interactive Selector: {total_pages} pages found.")
    print("Controls: [n/Right/Wheel] Nav, [Enter] Select, [Esc] Cancel")
    
    while True:
        rendered_page = render_page(doc, current_page, zoom=view_zoom)
        
        if rendered_page is None:
            print(f"Error rendering page {current_page}")
            rendered_page = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(rendered_page, f"Error rendering page {current_page}", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Draw UI Overlay
        display_img = rendered_page.copy()
        
        # Header Info
        header_height = 40
        # Check if we should grow the image to fit header? 
        # Easier to just draw on top for now.
        cv2.rectangle(display_img, (0, 0), (display_img.shape[1], header_height), (50, 50, 50), -1)
        
        info_text = f"Page {current_page + 1} / {total_pages} - Press ENTER to Process"
        cv2.putText(display_img, info_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_img)
        
        # Poll for keys frequently to handle smooth interactions
        key = cv2.waitKey(50)
        
        # Handle Scroll
        if params['scroll_delta'] != 0:
            if params['scroll_delta'] > 0: # Scroll Up -> Prev
                 if current_page > 0: current_page -= 1
            else: # Scroll Down -> Next
                 if current_page < total_pages - 1: current_page += 1
            params['scroll_delta'] = 0
            continue # Re-render immediately

        if key != -1:
            if key == 27: # Esc
                cv2.destroyWindow(window_name)
                return None
                
            elif key == 13: # Enter
                cv2.destroyWindow(window_name)
                return current_page
                
            elif key == ord('n') or key == 83: 
                if current_page < total_pages - 1:
                    current_page += 1
                    
            elif key == ord('b') or key == 81: 
                if current_page > 0:
                    current_page -= 1
