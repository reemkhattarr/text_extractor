import cv2
import argparse
import os
import string
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Try importing PDF loader logic
try:
    from pdf_loader import load_pdf, render_page, render_clip
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pdf_loader not found or pymupdf missing. PDF features will work partially or fail.")

# Try importing pytesseract
try:
    import pytesseract
    # Set explicit path if needed (Windows often needs this)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("pytesseract not found. Autosuggestion disabled.")

class TemplateTrainer:
    def __init__(self, output_dir="templates"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # State
        self.pdf_doc = None
        self.last_smart_select_time = 0 # Debounce
        self.pdf_path = None
        self.current_page = 0
        self.total_pages = 0
        
        self.original_page_img = None # The clean render of the page (High detail base)
        
        # Viewport State
        self.view_scale = 1.0 # Display zoom level
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.min_scale = 0.1 # Calc on load
        
        self.is_panning = False
        self.pan_start = (0, 0)
        
        # Selection State
        self.selecting = False
        self.selection_start = None # (img_x, img_y) in IMAGE coordinates
        self.selection_current = None 
        self.confirmed_rect = None # (x, y, w, h) in IMAGE coordinates
        
        # Session Data
        self.session_chars = set()
        self.existing_files = set()
        self.refresh_file_list()

        # UI Config
        self.window_name = "Template Trainer"
        self.sidebar_width = 300
        self.header_height = 60
        self.min_height = 800
        self.input_char = "" # Buffer for text box input
        self.selection_mode = "RECT" # "RECT" or "SELECT"
        
        # Cursor State
        self.cursor_visible = True
        self.last_blink_time = 0
        
        # Mouse Callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

# ... inside draw_sidebar ...

        # --- Input Box Logic ---
        input_y_start = 120
        
        if self.confirmed_rect:
            # Active Input Box
            box_x, box_y = 20, 110
            box_w, box_h = 260, 40
            
            # Highlight Box
            cv2.rectangle(canvas, (box_x, box_y), (box_x+box_w, box_y+box_h), (255, 255, 255), -1)
            cv2.rectangle(canvas, (box_x, box_y), (box_x+box_w, box_y+box_h), (0, 0, 0), 2)
            
            # Blink Cursor
            import time
            current_time = time.time()
            if current_time - self.last_blink_time > 0.5: # Blink every 500ms
                self.cursor_visible = not self.cursor_visible
                self.last_blink_time = current_time
            
            cursor_char = "|" if self.cursor_visible else " "
            
            # Text Cursor/Content
            prompt = f"Key: {self.input_char}{cursor_char}" 
            cv2.putText(canvas, prompt, (box_x + 10, box_y + 28), font, 0.6, (0, 0, 0), 1)
            cv2.putText(canvas, "Type A-Z, 0-9, then ENTER", (box_x, box_y + 60), font, 0.4, (100, 100, 100), 1)
            
            input_y_start = 190 # Push lists down
        else:
            self.cursor_visible = True # Reset when not active



    def predict_char(self, roi):
        """Uses Tesseract to predict the character in the ROI, checking multiple rotations."""
        if not PYTESSERACT_AVAILABLE or roi is None or roi.size == 0:
            return ""
            
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # PSM 10 = Treat the image as a single character
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        best_char = ""
        
        # 0 degrees
        try:
             text = pytesseract.image_to_string(thresh, config=config)
             char = text.strip().upper()
             if len(char) == 1 and (char.isalnum()):
                 # If valid upright, prefer it immediately, unless we want to be super robust?
                 # Usually if upright is recognizable, it's correct.
                 print(f"OCR Prediction (0 deg): {char}")
                 return char
        except: pass

        # Check Rotations: 90 (CW), 180, 270 (CCW)
        # 90 deg = Rotate 90 CW (which is ROTATE_90_CLOCKWISE)
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, "90 deg"),
            (cv2.ROTATE_180, "180 deg"),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, "270 deg")
        ]
        
        for rot_code, label in rotations:
            try:
                rotated = cv2.rotate(thresh, rot_code)
                text = pytesseract.image_to_string(rotated, config=config)
                char = text.strip().upper()
                if len(char) == 1 and (char.isalnum()):
                    print(f"OCR Prediction ({label}): {char}")
                    return char
            except: pass
            
        return ""

    def refresh_file_list(self):
        if os.path.exists(self.output_dir):
            files = os.listdir(self.output_dir)
            self.existing_files = {f.split('.')[0] for f in files if f.endswith('.png')}
    
    def load_pdf_dialog(self):
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.load_pdf(file_path)
        root.destroy()
            
    def load_pdf(self, path):
        if not PDF_AVAILABLE:
            print("PDF Support unavailable.")
            return

        print(f"Loading {path}...")
        self.pdf_doc = load_pdf(path)
        if self.pdf_doc:
            self.pdf_path = path
            self.total_pages = len(self.pdf_doc)
            self.current_page = 0
            self.update_page_render()
        else:
            print("Failed to load PDF.")

    def update_page_render(self):
        if not self.pdf_doc:
            return
            
        # Render high-res base
        # Ensure base zoom is high enough to capture detail, 
        # but handled efficiently.
        base_zoom = 4.0 
        print(f"Rendering page {self.current_page+1} at zoom {base_zoom}...")
        self.original_page_img = render_page(self.pdf_doc, self.current_page, zoom=base_zoom)
        
        # Reset Selection
        self.confirmed_rect = None
        self.selection_start = None
        
        # Calc Zoom-to-Fit
        if self.original_page_img is not None:
            ph, pw = self.original_page_img.shape[:2]
            
            # Viewport size (Approximate, simplified calc)
            # We assume a standard window height or current window height if possible?
            view_w = 1200 # Fixed canvas width
            view_h = 900 - self.header_height
            
            scale_w = view_w / pw
            scale_h = view_h / ph
            
            # Fit whole page
            self.min_scale = min(scale_w, scale_h) * 0.95 # slightly smaller to ensure margin?
            self.view_scale = self.min_scale 
            
            # Center it
            # Offset = (ViewSize - ImageSize*Scale) / 2
            self.view_offset_x = (view_w - pw * self.view_scale) / 2
            self.view_offset_y = (view_h - ph * self.view_scale) / 2
            
    # --- Coordinate Transforms ---
    def screen_to_image(self, sx, sy):
        ix = (sx - self.view_offset_x) / self.view_scale
        iy = (sy - self.view_offset_y) / self.view_scale
        return int(ix), int(iy)

    def image_to_screen(self, ix, iy):
        sx = ix * self.view_scale + self.view_offset_x
        sy = iy * self.view_scale + self.view_offset_y
        return int(sx), int(sy)

    def find_nearest_non_white(self, img, x, y, max_dist=50):
        # Spiral or localized search
        h, w = img.shape[:2]
        
        # Check center first
        if np.mean(img[y, x]) < 200: # Assuming white is > 200
            return x, y
            
        for d in range(1, max_dist, 2):
            x1, x2 = max(0, x-d), min(w-1, x+d)
            y1, y2 = max(0, y-d), min(h-1, y+d)
            
            # Extract ROI
            roi = img[y1:y2+1, x1:x2+1]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Find dark pixels
            dark_pixels = np.where(gray_roi < 200) # Threshold for "not white"
            
            if len(dark_pixels[0]) > 0:
                # Found closest
                # Calculate absolute positions
                points = []
                for i in range(len(dark_pixels[0])):
                     ly, lx = dark_pixels[0][i], dark_pixels[1][i]
                     abs_x, abs_y = x1 + lx, y1 + ly
                     dist = (abs_x - x)**2 + (abs_y - y)**2
                     points.append((dist, abs_x, abs_y))
                
                points.sort(key=lambda p: p[0])
                return points[0][1], points[0][2]
        return None

    def do_smart_selection(self, img_x, img_y):
        if self.original_page_img is None: return
        
        h, w = self.original_page_img.shape[:2]
        if not (0 <= img_x < w and 0 <= img_y < h): return
        
        target = (img_x, img_y)
        
        # 1. Check if clicking on white space
        pixel_val = np.mean(self.original_page_img[img_y, img_x])
        if pixel_val > 220: # White-ish
             res = self.find_nearest_non_white(self.original_page_img, img_x, img_y)
             if res:
                 target = res
             else:
                 print("No dark pixel found nearby.")
                 return
        
        # 2. Flood Fill with Retry Logic
        tolerances = [10, 15, 20, 30, 40]
        found_selection = False
        
        for tol in tolerances:
            mask = np.zeros((h+2, w+2), np.uint8)
            loDiff = (tol, tol, tol)
            upDiff = (tol, tol, tol)
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            
            try:
                 retval, _, mask, rect = cv2.floodFill(
                    self.original_page_img, 
                    mask, 
                    target, 
                    newVal=(255, 0, 0), # ignored for mask only
                    loDiff=loDiff, 
                    upDiff=upDiff, 
                    flags=flags
                )
                 
                 if rect[2] > 2 and rect[3] > 2:
                     # Expand slightly to capture edges (user allowed padding)
                     rx, ry, rw, rh = rect
                     pad = 2
                     nx = max(0, rx - pad)
                     ny = max(0, ry - pad)
                     nw = rw + (rx - nx) + pad # add left delta + right pad
                     nh = rh + (ry - ny) + pad
                     
                     self.confirmed_rect = (nx, ny, nw, nh)
                     print(f"Smart Selected (Tol {tol}): {self.confirmed_rect}")
                     
                     # Autosuggest using PADDED rect
                     roi = self.original_page_img[ny:ny+nh, nx:nx+nw]
                     self.input_char = self.predict_char(roi)
                     
                     found_selection = True
                     break # Stop if we found a good one
                     
            except Exception as e:
                print(f"Flood fill failed at tol {tol}: {e}")
                
        if not found_selection:
             print("Selection too small (try clicking darker center or check zoom).")
             
        self.last_smart_select_time = time.time()


    def mouse_callback(self, event, x, y, flags, param):
        # 1. UI Buttons
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pdf_doc is None: # Start Screen
                if x > self.sidebar_width: 
                    self.load_pdf_dialog()
                    return
            
            # Nav Buttons
            if self.pdf_doc and y <= self.header_height:
                sx = self.sidebar_width
                if sx + 10 <= x <= sx + 90 and 10 <= y <= 50:
                    self.change_page(-1)
                    return
                if sx + 100 <= x <= sx + 180 and 10 <= y <= 50:
                    self.change_page(1)
                    return
            
            # Mode Button
            if x <= self.sidebar_width:
                 if 20 <= x <= 180 and 80 <= y <= 95: # Click area
                     if self.selection_mode == "RECT": self.selection_mode = "SELECT"
                     else: self.selection_mode = "RECT"
                     
                     # Clear selection on mode switch
                     self.confirmed_rect = None
                     self.input_char = ""
                     self.selection_start = None
                     self.selection_current = None
                     self.selecting = False
                     print("Selection cleared due to mode switch.")
                     
                     print(f"Mode switched to {self.selection_mode}")
                     return
        
        # 2. Viewport Interactions
        if self.pdf_doc and x > self.sidebar_width and y > self.header_height:
            vx = x - self.sidebar_width
            vy = y - self.header_height
            
            # Pan
            if event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_RBUTTONDOWN):
                self.is_panning = True
                self.pan_start = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
                dx = x - self.pan_start[0]
                dy = y - self.pan_start[1]
                self.view_offset_x += dx
                self.view_offset_y += dy
                self.pan_start = (x, y)
            
            elif event == cv2.EVENT_MBUTTONUP or (event == cv2.EVENT_RBUTTONUP):
                self.is_panning = False
            
            # Zoom
            elif event == cv2.EVENT_MOUSEWHEEL:
                zoom_factor = 1.1 if flags > 0 else 0.9
                
                mx_img, my_img = self.screen_to_image(vx, vy)
                
                new_scale = self.view_scale * zoom_factor
                # Limit minimum zoom to fit
                new_scale = max(self.min_scale, min(new_scale, 5.0))
                
                self.view_offset_x = vx - mx_img * new_scale
                self.view_offset_y = vy - my_img * new_scale
                self.view_scale = new_scale
            
            # Selection
            elif event == cv2.EVENT_LBUTTONDOWN:
                if self.selection_mode == "RECT":
                    # Always allow starting a new selection (discards old confirmed)
                    self.confirmed_rect = None
                    self.input_char = ""
                    
                    self.selecting = True
                    img_pos = self.screen_to_image(vx, vy)
                    self.selection_start = img_pos
                    self.selection_current = img_pos
            
            elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
                img_pos = self.screen_to_image(vx, vy)
                self.selection_current = img_pos
            
            elif event == cv2.EVENT_LBUTTONUP: 
                if self.selecting:
                    self.selecting = False
                    # Auto-confirm logic
                    if self.selection_start and self.selection_current:
                        x1, y1 = self.selection_start
                        x2, y2 = self.selection_current
                        rx, ry = min(x1, x2), min(y1, y2)
                        rw, rh = abs(x1 - x2), abs(y1 - y2)
                        
                        if rw > 2 and rh > 2:
                            self.confirmed_rect = (rx, ry, rw, rh)
                            print(f"Auto-selected: {self.confirmed_rect}")
                            
                            # Autosuggest immediately
                            if self.original_page_img is not None:
                                safe_rx = max(0, rx)
                                safe_ry = max(0, ry)
                                safe_w = min(self.original_page_img.shape[1] - safe_rx, rw)
                                safe_h = min(self.original_page_img.shape[0] - safe_ry, rh)
                                
                                if safe_w > 0 and safe_h > 0:
                                    roi = self.original_page_img[safe_ry:safe_ry+safe_h, safe_rx:safe_rx+safe_w]
                                    self.input_char = self.predict_char(roi)

                elif self.selection_mode == "SELECT":
                    if not self.is_panning:
                         # Debounce: Ignore clicks too close to last processing end
                         if time.time() - self.last_smart_select_time < 0.5:
                             return

                         # Discard previous if exists (User requested no auto-save)
                         if self.confirmed_rect:
                             self.confirmed_rect = None
                             self.input_char = ""
                             
                         # Start new selection
                         img_pos = self.screen_to_image(vx, vy)
                         self.do_smart_selection(img_pos[0], img_pos[1])

    def change_page(self, delta):
        if not self.pdf_doc: return
        new_page = self.current_page + delta
        if 0 <= new_page < self.total_pages:
            self.current_page = new_page
            self.update_page_render()

    def draw_sidebar(self, canvas):
        # Sidebar Background
        sidebar = canvas[:, :self.sidebar_width]
        sidebar[:] = (245, 245, 245)
        cv2.line(canvas, (self.sidebar_width, 0), (self.sidebar_width, canvas.shape[0]), (200, 200, 200), 1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(canvas, "Templates", (20, 40), font, 0.6, (50, 50, 50), 1)
        
        # Legend
        cv2.rectangle(canvas, (20, 60), (35, 75), (200, 230, 200), -1) 
        cv2.putText(canvas, "Done", (45, 72), font, 0.4, (100, 100, 100), 1)
        cv2.rectangle(canvas, (90, 60), (105, 75), (220, 220, 220), -1)
        cv2.putText(canvas, "To Add", (115, 72), font, 0.4, (100, 100, 100), 1)
        
        # Mode Toggle
        mode_btn_col = (200, 200, 240) if self.selection_mode == "SELECT" else (220, 220, 220)
        cv2.rectangle(canvas, (20, 80), (180, 95), mode_btn_col, -1)
        cv2.rectangle(canvas, (20, 80), (180, 95), (150, 150, 150), 1)
        mode_text = f"Mode: {self.selection_mode}"
        cv2.putText(canvas, mode_text, (30, 91), font, 0.4, (0, 0, 0), 1)

        # --- Input Box Logic ---
        input_y_start = 120
        
        if self.confirmed_rect:
            # Active Input Box
            box_x, box_y = 20, 110
            box_w, box_h = 260, 40
            
            # Highlight Box
            cv2.rectangle(canvas, (box_x, box_y), (box_x+box_w, box_y+box_h), (255, 255, 255), -1)
            cv2.rectangle(canvas, (box_x, box_y), (box_x+box_w, box_y+box_h), (0, 0, 0), 2)
            
            # Blink Cursor
            import time
            current_time = time.time()
            if current_time - self.last_blink_time > 0.5: # Blink every 500ms
                self.cursor_visible = not self.cursor_visible
                self.last_blink_time = current_time
            
            cursor_char = "|" if self.cursor_visible else " "
            
            # Text Cursor/Content
            prompt = f"Key: {self.input_char}{cursor_char}" 
            cv2.putText(canvas, prompt, (box_x + 10, box_y + 28), font, 0.6, (0, 0, 0), 1)
            cv2.putText(canvas, "Type A-Z, 0-9, then ENTER", (box_x, box_y + 60), font, 0.4, (100, 100, 100), 1)
            
            input_y_start = 190 # Push lists down
        else:
            self.cursor_visible = True # Reset
            pass

        # Lists
        digits = string.digits
        letters = string.ascii_uppercase
        
        cell_size = 30
        cols_per_row = 6
        start_x = 20
        
        def draw_grid(chars, offset_y):
            for i, char in enumerate(chars):
                row = i // cols_per_row
                col = i % cols_per_row
                xx = start_x + col * (cell_size + 5)
                yy = offset_y + row * (cell_size + 5)
                
                bg = (255, 255, 255)
                fg = (180, 180, 180)
                border = (220, 220, 220)
                
                if char in self.session_chars:
                    bg = (200, 230, 200)
                    fg = (0, 60, 0)
                    border = (150, 200, 150)
                elif char in self.existing_files:
                    bg = (220, 220, 220)
                    fg = (20, 20, 20)
                    border = (200, 200, 200)
                
                cv2.rectangle(canvas, (xx, yy), (xx+cell_size, yy+cell_size), bg, -1)
                cv2.rectangle(canvas, (xx, yy), (xx+cell_size, yy+cell_size), border, 1)
                
                (tw, th), _ = cv2.getTextSize(char, font, 0.4, 1)
                tx = xx + (cell_size - tw) // 2
                ty = yy + (cell_size + th) // 2
                cv2.putText(canvas, char, (tx, ty), font, 0.4, fg, 1)
            return offset_y + (len(chars) // cols_per_row + 1) * (cell_size + 5) + 20

        next_y = draw_grid(digits, input_y_start)
        draw_grid(letters, next_y)
        
    def compose_ui(self):
        canvas_h = 900
        canvas_w = 1200 + self.sidebar_width
        
        # Re-allocate canvas
        self.canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        self.draw_sidebar(self.canvas)
        
        if self.pdf_doc is None:
            # Start Screen
            msg = "No PDF Loaded"
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(msg, font, 0.8, 1)
            cx = self.sidebar_width + (1200 - tw)//2
            cy = canvas_h // 2
            cv2.putText(self.canvas, msg, (cx, cy), font, 0.8, (150, 150, 150), 1)
            
            btn_w, btn_h = 200, 50
            bx = self.sidebar_width + (1200 - btn_w)//2
            by = cy + 40
            cv2.rectangle(self.canvas, (bx, by), (bx+btn_w, by+btn_h), (230, 230, 230), -1)
            cv2.rectangle(self.canvas, (bx, by), (bx+btn_w, by+btn_h), (180, 180, 180), 1)
            label = "OPEN PDF"
            (LW, LH), _ = cv2.getTextSize(label, font, 0.6, 1)
            lx = bx + (btn_w - LW)//2
            ly = by + (btn_h + LH)//2
            cv2.putText(self.canvas, label, (lx, ly), font, 0.6, (50, 50, 50), 1)
            
        else:
            # Header
            header_bg = (240, 240, 240)
            cv2.rectangle(self.canvas, (self.sidebar_width, 0), (canvas_w, self.header_height), header_bg, -1)
            
            sx = self.sidebar_width
            # Prev
            cv2.rectangle(self.canvas, (sx + 10, 10), (sx + 90, 50), (220, 220, 220), -1)
            cv2.putText(self.canvas, "< PREV", (sx + 25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # Next
            cv2.rectangle(self.canvas, (sx + 100, 10), (sx + 180, 50), (220, 220, 220), -1)
            cv2.putText(self.canvas, "NEXT >", (sx + 115, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            p_info = f"Page {self.current_page + 1}/{self.total_pages}"
            cv2.putText(self.canvas, p_info, (sx + 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
            
            if self.confirmed_rect:
                instr = "Type Character (Sidebar) or Drag New Box to Discard"
                col = (0, 0, 150)
            elif self.selection_start:
                 instr = "Release to Confirm"
                 col = (0, 100, 0)
            else:
                instr = "Drag Left-Mouse to Select. Right-Mouse to Pan. Wheel to Zoom."
                col = (100, 100, 100)
            cv2.putText(self.canvas, instr, (sx + 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # Draw Viewport
            if self.original_page_img is not None:
                view_w = canvas_w - self.sidebar_width
                view_h = canvas_h - self.header_height
                
                M = np.float32([
                    [self.view_scale, 0, self.view_offset_x],
                    [0, self.view_scale, self.view_offset_y]
                ])
                
                viewport_img = cv2.warpAffine(self.original_page_img, M, (view_w, view_h), borderValue=(200, 200, 200))
                
                # Selection Logic on Viewport
                if self.selecting and self.selection_start and self.selection_current:
                    s_sx, s_sy = self.image_to_screen(*self.selection_start)
                    c_sx, c_sy = self.image_to_screen(*self.selection_current)
                    cv2.rectangle(viewport_img, (s_sx, s_sy), (c_sx, c_sy), (0, 0, 255), 1)
                elif self.selection_start and self.selection_current and not self.confirmed_rect:
                    s_sx, s_sy = self.image_to_screen(*self.selection_start)
                    c_sx, c_sy = self.image_to_screen(*self.selection_current)
                    cv2.rectangle(viewport_img, (s_sx, s_sy), (c_sx, c_sy), (0, 200, 0), 2)
                    
                if self.confirmed_rect:
                    rx, ry, rw, rh = self.confirmed_rect
                    p1 = self.image_to_screen(rx, ry)
                    p2 = self.image_to_screen(rx+rw, ry+rh)
                    cv2.rectangle(viewport_img, p1, p2, (0, 255, 0), 2)
                    # Removed '?' drawing here as requested
                
                self.canvas[self.header_height:, self.sidebar_width:] = viewport_img

        return self.canvas

    def save_template(self, char_key):
        if not self.confirmed_rect or not char_key: return
            
        print(f"Saving template for '{char_key}'...")
        rx, ry, rw, rh = self.confirmed_rect
        final_crop = None
        
        if self.pdf_doc:
            try:
                base_zoom = 4.0
                pdf_x0 = rx / base_zoom
                pdf_y0 = ry / base_zoom
                pdf_x1 = (rx + rw) / base_zoom
                pdf_y1 = (ry + rh) / base_zoom
                
                capture_zoom = 24.0
                final_crop = render_clip(self.pdf_doc, self.current_page, (pdf_x0, pdf_y0, pdf_x1, pdf_y1), zoom=capture_zoom)
            except Exception as e:
                print(f"High res extraction failed: {e}")
        
        if final_crop is None or final_crop.size == 0:
            h, w = self.original_page_img.shape[:2]
            y1, y2 = max(0, int(ry)), min(h, int(ry+rh))
            x1, x2 = max(0, int(rx)), min(w, int(rx+rw))
            final_crop = self.original_page_img[y1:y2, x1:x2]

        if final_crop is not None and final_crop.size > 0:
            gray = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
            safe_char = char_key
            if char_key == ".": safe_char = "dot"
            if char_key == "/": safe_char = "slash"
            path = os.path.join(self.output_dir, f"{safe_char}.png")
            cv2.imwrite(path, gray)
            
            self.session_chars.add(char_key)
            self.refresh_file_list() 
            self.confirmed_rect = None
            self.selection_start = None 
            self.selection_current = None
            self.input_char = "" # Reset buffer
            print(f"Saved {path}")

    def run(self):
        while True:
            ui = self.compose_ui()
            cv2.imshow(self.window_name, ui)
            key = cv2.waitKey(20) & 0xFF
            
            if key == 27: # Esc
                if self.confirmed_rect:
                    self.confirmed_rect = None
                    self.input_char = ""
                    print("Selection cleared.")
                elif self.selecting:
                     self.selecting = False
                else: 
                     break
            
            elif key == 13: # Enter
                if self.confirmed_rect:
                    # Confirm Label
                    if len(self.input_char) > 0:
                        self.save_template(self.input_char)
            
            elif key == 8: # Backspace
                self.input_char = self.input_char[:-1]
                
            elif self.confirmed_rect:
                # Typing Mode
                char = None
                if ord('0') <= key <= ord('9'): char = chr(key)
                elif ord('a') <= key <= ord('z'): char = chr(key).upper()
                elif ord('A') <= key <= ord('Z'): char = chr(key)
                
                if char:
                    # Allow only single char logic basically
                    self.input_char = char 
            
            else:
                if key == ord('n'): self.change_page(1)
                if key == ord('b'): self.change_page(-1)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", help="Optional initial PDF path")
    args = parser.parse_args()
    
    app = TemplateTrainer()
    if args.image_path:
        app.load_pdf(args.image_path)
    
    app.run()
