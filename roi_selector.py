import cv2
import numpy as np

# Interaction Modes
MODE_NONE = 0
MODE_CREATE = 1
MODE_MOVE = 2
MODE_RESIZE_TL = 3
MODE_RESIZE_TR = 4
MODE_RESIZE_BL = 5
MODE_RESIZE_BR = 6

HANDLE_RADIUS = 5

def get_rotation_params(image_shape, angle):
    """
    Calculates the rotation matrix and new dimensions for a center-rotated image.
    """
    h, w = image_shape[:2]
    
    if angle == 0:
        return None, w, h
        
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return M, new_w, new_h

class ZoomableROISelector:
    def __init__(self, image, window_name="Select ROI"):
        self.original_image = image
        self.window_name = window_name
        self.roi = None
        
        # View state
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.rotation_angle = 0  # 0, 90, 180, 270...
        
        # Interaction state
        self.mode = MODE_NONE
        self.active_mode = MODE_NONE # Mode while dragging
        
        self.is_dragging_pan = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.initial_rect = None # For move/resize delta calc
        self.initial_start_pt = None # For CREATE mode
        
        self.selection_rect = None # (x, y, w, h) in ROTATED image coordinates
        
        self.running = True
        self.confirmed = False
        self.action = None
        self.current_view_image = None

    def update_image(self, new_image):
        """Updates the image being displayed (e.g. on page change)."""
        self.original_image = new_image
        self.roi = None
        self.selection_rect = None
        self.update_display()

    def get_rotated_image(self):
        """Returns the image rotated by self.rotation_angle."""
        if self.rotation_angle == 0:
            return self.original_image.copy()
        
        M, new_w, new_h = get_rotation_params(self.original_image.shape, self.rotation_angle)
        
        rotated = cv2.warpAffine(self.original_image, M, (new_w, new_h))
        return rotated

    def image_to_view(self, ix, iy):
        vx = int(ix * self.scale + self.offset_x)
        vy = int(iy * self.scale + self.offset_y)
        return vx, vy

    def view_to_image(self, vx, vy):
        ix = (vx - self.offset_x) / self.scale
        iy = (vy - self.offset_y) / self.scale
        return int(ix), int(iy)

    def get_hit_mode(self, vx, vy):
        if not self.selection_rect:
            return MODE_CREATE
            
        sx, sy, sw, sh = self.selection_rect
        
        # Get corner points in view
        tl = self.image_to_view(sx, sy)
        tr = self.image_to_view(sx + sw, sy)
        bl = self.image_to_view(sx, sy + sh)
        br = self.image_to_view(sx + sw, sy + sh)
        
        # Check handles
        def dist(p1, p2): return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        if dist((vx, vy), tl) <= HANDLE_RADIUS + 2: return MODE_RESIZE_TL
        if dist((vx, vy), tr) <= HANDLE_RADIUS + 2: return MODE_RESIZE_TR
        if dist((vx, vy), bl) <= HANDLE_RADIUS + 2: return MODE_RESIZE_BL
        if dist((vx, vy), br) <= HANDLE_RADIUS + 2: return MODE_RESIZE_BR
        
        # Check inside
        # Simple AABB check in view coords? 
        # Since rotation is handled by image buffer, the rect is axis aligned in image enum View.
        # So it stays axis aligned in view.
        if tl[0] < vx < br[0] and tl[1] < vy < br[1]:
            return MODE_MOVE
            
        return MODE_CREATE

    def update_display(self):
        rotated_img = self.get_rotated_image()
        h, w = rotated_img.shape[:2]
        
        # Warp Affine for Display
        M = np.float32([
            [self.scale, 0, self.offset_x],
            [0, self.scale, self.offset_y]
        ])
        
        view_w, view_h = 1200, 800
        disp_img = cv2.warpAffine(rotated_img, M, (view_w, view_h))
        
        # Draw selection rectangle and handles
        if self.selection_rect:
            rx, ry, rw, rh = self.selection_rect
            
            p1 = self.image_to_view(rx, ry)
            p2 = self.image_to_view(rx + rw, ry + rh)
            
            # Draw rect
            cv2.rectangle(disp_img, p1, p2, (0, 255, 0), 2)
            
            # Draw handles
            # p1 is TL, p2 is BR
            tl = p1
            tr = (p2[0], p1[1])
            bl = (p1[0], p2[1])
            br = p2
            
            handles = [tl, tr, bl, br]
            for pt in handles:
                cv2.circle(disp_img, pt, HANDLE_RADIUS, (0, 0, 255), -1)
                
        self.current_view_image = disp_img
        cv2.imshow(self.window_name, disp_img)

    def clamp_rect(self, rect, max_w, max_h):
        x, y, w, h = rect
        # Normalize (negative width/height handling)
        if w < 0: 
            x += w
            w = abs(w)
        if h < 0:
            y += h
            h = abs(h)
            
        return (x, y, w, h)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine mode
            self.active_mode = self.get_hit_mode(x, y)
            self.drag_start_x = x
            self.drag_start_y = y
            self.initial_rect = self.selection_rect
            
            if self.active_mode == MODE_CREATE:
                self.selection_rect = None # Start fresh
                # Initial rect is just this point
                ix, iy = self.view_to_image(x, y)
                self.initial_start_pt = (ix, iy) # Store start point in image coords
            
            self.update_display()

        elif event == cv2.EVENT_MOUSEMOVE:
            ix, iy = self.view_to_image(x, y)
            
            if self.is_dragging_pan:
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                self.offset_x += dx
                self.offset_y += dy
                self.pan_start_x = x
                self.pan_start_y = y
                self.update_display()
                
            elif self.active_mode != MODE_NONE:
                # Handle Interactions
                if self.active_mode == MODE_CREATE:
                    sx, sy = self.initial_start_pt
                    w = ix - sx
                    h = iy - sy
                    self.selection_rect = self.clamp_rect((sx, sy, w, h), 0, 0)
                    
                elif self.active_mode == MODE_MOVE and self.initial_rect:
                    rx, ry, rw, rh = self.initial_rect
                    # Delta in image coords
                    start_ix, start_iy = self.view_to_image(self.drag_start_x, self.drag_start_y)
                    dx = ix - start_ix
                    dy = iy - start_iy
                    self.selection_rect = (rx + dx, ry + dy, rw, rh)
                    
                elif self.initial_rect: # Resizing
                    rx, ry, rw, rh = self.initial_rect
                    
                    if self.active_mode == MODE_RESIZE_BR:
                        # Top-Left is fixed: (rx, ry)
                        self.selection_rect = self.clamp_rect((rx, ry, ix - rx, iy - ry), 0, 0)
                    elif self.active_mode == MODE_RESIZE_TL:
                        # Bottom-Right is fixed: (rx+rw, ry+rh)
                        br_x = rx + rw
                        br_y = ry + rh
                        self.selection_rect = self.clamp_rect((ix, iy, br_x - ix, br_y - iy), 0, 0)
                    elif self.active_mode == MODE_RESIZE_TR:
                        # Bottom-Left is fixed: (rx, ry+rh)
                        # New TR is (ix, iy)
                        # Top-Left X = rx, Top-Left Y = iy
                        # W = ix - rx, H = (ry+rh) - iy
                        self.selection_rect = self.clamp_rect((rx, iy, ix - rx, ry + rh - iy), 0, 0)
                    elif self.active_mode == MODE_RESIZE_BL:
                        # Top-Right is fixed: (rx+rw, ry)
                        # Top-Left X = ix, Top-Left Y = ry
                        # W = (rx+rw) - ix, H = iy - ry
                        self.selection_rect = self.clamp_rect((ix, ry, rx + rw - ix, iy - ry), 0, 0)

                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.active_mode = MODE_NONE
            # Normalize rect (no negative w/h)
            if self.selection_rect:
                self.selection_rect = self.clamp_rect(self.selection_rect, 0, 0)
            self.update_display()

        elif event == cv2.EVENT_MBUTTONDOWN: # Middle button for pan
            self.is_dragging_pan = True
            self.pan_start_x = x
            self.pan_start_y = y

        elif event == cv2.EVENT_MBUTTONUP:
            self.is_dragging_pan = False
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1 if flags > 0 else 0.9
            img_x = (x - self.offset_x) / self.scale
            img_y = (y - self.offset_y) / self.scale
            self.scale *= zoom_factor
            self.offset_x = x - img_x * self.scale
            self.offset_y = y - img_y * self.scale
            self.update_display()


    def select_roi(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while self.running:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC
                self.running = False
                self.roi = None
                break
            elif key == 32 or key == 13: # SPACE or ENTER
                if self.selection_rect:
                    self.confirmed = True
                    self.running = False
                    
                    r_img = self.get_rotated_image()
                    x, y, w, h = self.selection_rect
                    
                    # Strict Bounds Intersection (Fixing the wrapping bug)
                    h_img, w_img = r_img.shape[:2]
                    
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w_img, x + w)
                    y2 = min(h_img, y + h)
                    
                    # Recalculate w, h
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w > 0 and h > 0:
                        self.roi = r_img[y1:y1+h, x1:x1+w]
                    else:
                        print("Invalid selection (Out of bounds).")
                        self.running = True
                        continue
                else:
                    # If no selection, treat as confirming the view (e.g. for page selection)
                    self.action = 'confirm_view'
                    self.running = False
 

        # So it stays axis aligned in view.
        if tl[0] < vx < br[0] and tl[1] < vy < br[1]:
            return MODE_MOVE
            
        return MODE_CREATE

    def update_display(self):
        rotated_img = self.get_rotated_image()
        h, w = rotated_img.shape[:2]
        
        # Warp Affine for Display
        M = np.float32([
            [self.scale, 0, self.offset_x],
            [0, self.scale, self.offset_y]
        ])
        
        view_w, view_h = 1200, 800
        disp_img = cv2.warpAffine(rotated_img, M, (view_w, view_h))
        
        # Draw selection rectangle and handles
        if self.selection_rect:
            rx, ry, rw, rh = self.selection_rect
            
            p1 = self.image_to_view(rx, ry)
            p2 = self.image_to_view(rx + rw, ry + rh)
            
            # Draw rect
            cv2.rectangle(disp_img, p1, p2, (0, 255, 0), 2)
            
            # Draw handles
            # p1 is TL, p2 is BR
            tl = p1
            tr = (p2[0], p1[1])
            bl = (p1[0], p2[1])
            br = p2
            
            handles = [tl, tr, bl, br]
            for pt in handles:
                cv2.circle(disp_img, pt, HANDLE_RADIUS, (0, 0, 255), -1)
                
        self.current_view_image = disp_img
        cv2.imshow(self.window_name, disp_img)

    def clamp_rect(self, rect, max_w, max_h):
        x, y, w, h = rect
        # Normalize (negative width/height handling)
        if w < 0: 
            x += w
            w = abs(w)
        if h < 0:
            y += h
            h = abs(h)
            
        return (x, y, w, h)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine mode
            self.active_mode = self.get_hit_mode(x, y)
            self.drag_start_x = x
            self.drag_start_y = y
            self.initial_rect = self.selection_rect
            
            if self.active_mode == MODE_CREATE:
                self.selection_rect = None # Start fresh
                # Initial rect is just this point
                ix, iy = self.view_to_image(x, y)
                self.initial_start_pt = (ix, iy) # Store start point in image coords
            
            self.update_display()

        elif event == cv2.EVENT_MOUSEMOVE:
            ix, iy = self.view_to_image(x, y)
            
            if self.is_dragging_pan:
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                self.offset_x += dx
                self.offset_y += dy
                self.pan_start_x = x
                self.pan_start_y = y
                self.update_display()
                
            elif self.active_mode != MODE_NONE:
                # Handle Interactions
                if self.active_mode == MODE_CREATE:
                    sx, sy = self.initial_start_pt
                    w = ix - sx
                    h = iy - sy
                    self.selection_rect = self.clamp_rect((sx, sy, w, h), 0, 0)
                    
                elif self.active_mode == MODE_MOVE and self.initial_rect:
                    rx, ry, rw, rh = self.initial_rect
                    # Delta in image coords
                    start_ix, start_iy = self.view_to_image(self.drag_start_x, self.drag_start_y)
                    dx = ix - start_ix
                    dy = iy - start_iy
                    self.selection_rect = (rx + dx, ry + dy, rw, rh)
                    
                elif self.initial_rect: # Resizing
                    rx, ry, rw, rh = self.initial_rect
                    
                    if self.active_mode == MODE_RESIZE_BR:
                        # Top-Left is fixed: (rx, ry)
                        self.selection_rect = self.clamp_rect((rx, ry, ix - rx, iy - ry), 0, 0)
                    elif self.active_mode == MODE_RESIZE_TL:
                        # Bottom-Right is fixed: (rx+rw, ry+rh)
                        br_x = rx + rw
                        br_y = ry + rh
                        self.selection_rect = self.clamp_rect((ix, iy, br_x - ix, br_y - iy), 0, 0)
                    elif self.active_mode == MODE_RESIZE_TR:
                        # Bottom-Left is fixed: (rx, ry+rh)
                        # New TR is (ix, iy)
                        # Top-Left X = rx, Top-Left Y = iy
                        # W = ix - rx, H = (ry+rh) - iy
                        self.selection_rect = self.clamp_rect((rx, iy, ix - rx, ry + rh - iy), 0, 0)
                    elif self.active_mode == MODE_RESIZE_BL:
                        # Top-Right is fixed: (rx+rw, ry)
                        # Top-Left X = ix, Top-Left Y = ry
                        # W = (rx+rw) - ix, H = iy - ry
                        self.selection_rect = self.clamp_rect((ix, ry, rx + rw - ix, iy - ry), 0, 0)

                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.active_mode = MODE_NONE
            # Normalize rect (no negative w/h)
            if self.selection_rect:
                self.selection_rect = self.clamp_rect(self.selection_rect, 0, 0)
            self.update_display()

        elif event == cv2.EVENT_MBUTTONDOWN: # Middle button for pan
            self.is_dragging_pan = True
            self.pan_start_x = x
            self.pan_start_y = y

        elif event == cv2.EVENT_MBUTTONUP:
            self.is_dragging_pan = False
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1 if flags > 0 else 0.9
            img_x = (x - self.offset_x) / self.scale
            img_y = (y - self.offset_y) / self.scale
            self.scale *= zoom_factor
            self.offset_x = x - img_x * self.scale
            self.offset_y = y - img_y * self.scale
            self.update_display()


    def select_roi(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while self.running:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC
                self.running = False
                self.roi = None
                break
            elif key == 32 or key == 13: # SPACE or ENTER
                if self.selection_rect:
                    self.confirmed = True
                    self.running = False
                    
                    r_img = self.get_rotated_image()
                    x, y, w, h = self.selection_rect
                    
                    # Strict Bounds Intersection (Fixing the wrapping bug)
                    h_img, w_img = r_img.shape[:2]
                    
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w_img, x + w)
                    y2 = min(h_img, y + h)
                    
                    # Recalculate w, h
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w > 0 and h > 0:
                        self.roi = r_img[y1:y1+h, x1:x1+w]
                    else:
                        print("Invalid selection (Out of bounds).")
                        self.running = True
                        continue
                else:
                    # If no selection, treat as confirming the view (e.g. for page selection)
                    self.action = 'confirm_view'
                    self.running = False
            elif key == ord('q'):
                self.running = False
                self.roi = None
                break 
                
            elif key == ord('r'): # Rotate Clockwise
                self.rotation_angle = (self.rotation_angle - 45) % 360
                self.selection_rect = None # Reset selection on rotate
                self.update_display()
                
            elif key == ord('l'): 
                self.rotation_angle = (self.rotation_angle + 45) % 360
                self.selection_rect = None
                self.update_display()
            elif key == ord('p'):
                pass
            
            # Standardize navigation keys
            elif key == ord('n'):
                self.action = 'next_page'
                self.running = False
            elif key == ord('b'):
                self.action = 'prev_page'
                self.running = False
                
        cv2.destroyWindow(self.window_name)
        return self.roi, self.selection_rect, self.rotation_angle, self.confirmed, self.action

