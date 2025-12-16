import cv2
import numpy as np

def match_character(candidate_img, templates):
    """
    Matches a candidate image crop against all templates.
    Returns: (best_char, best_score)
    """
    best_score = -1
    best_char = "?"
    best_angle = 0
    
    h_cand, w_cand = candidate_img.shape[:2]
    if h_cand < 5 or w_cand < 2: 
        return None, 0.0

    # Define rotations to check
    # 0, 90 (counter clockwise), 180, 270
    rotations = [0, 90, 180, 270]
    
    for angle in rotations:
        # Rotate candidate
        if angle == 0:
            rotated_cand = candidate_img
        elif angle == 90:
            rotated_cand = cv2.rotate(candidate_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            rotated_cand = cv2.rotate(candidate_img, cv2.ROTATE_180)
        elif angle == 270:
            rotated_cand = cv2.rotate(candidate_img, cv2.ROTATE_90_CLOCKWISE)
            
        h_rot, w_rot = rotated_cand.shape[:2]
        
        for char, tmpl in templates.items():
            h_tmpl, w_tmpl = tmpl.shape[:2]
            if h_tmpl == 0: continue
            
            # Scale template to match rotated candidate height
            scale_factor = h_rot / float(h_tmpl)
            new_w = int(w_tmpl * scale_factor)
            new_h = h_rot 
            
            if new_w <= 0: continue
            
            # Simple width check (lax)
            if new_w > w_rot * 2.0: continue
            
            resized_tmpl = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            match_w = max(w_rot, new_w)
            match_h = max(h_rot, new_h)
            
            canvas = np.full((match_h, match_w), 255, dtype=np.uint8)
            y_off = (match_h - h_rot) // 2
            x_off = (match_w - w_rot) // 2
            canvas[y_off:y_off+h_rot, x_off:x_off+w_rot] = rotated_cand
            
            if resized_tmpl.shape[0] > match_h or resized_tmpl.shape[1] > match_w:
                continue
                
            res = cv2.matchTemplate(canvas, resized_tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_char = char
                best_angle = angle
            
    return best_char, best_score
