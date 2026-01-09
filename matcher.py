import cv2
import numpy as np

def match_character(candidate_img, templates, expected_scale=None, debug=False, debug_dir="debug_match"):
    """
    Matches a candidate image crop against all templates.
    Returns: (best_char, best_score)
    """
    import os
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    best_score = -1
    best_char = "?"
    best_angle = 0
    
    h_cand, w_cand = candidate_img.shape[:2]
    if h_cand < 5 or w_cand < 2: 
        if debug: print(f"Rejected candidate: too small {w_cand}x{h_cand}")
        return None, 0.0

    # Define rotations to check (0 to 315 in 45 deg steps)
    rotations = range(0, 360, 45)
    
    def rotate_img(image, angle):
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # General rotation for 45, 135, etc.
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # Use existing background color or white
        return cv2.warpAffine(image, M, (nW, nH), borderValue=255)

    for angle in rotations:
        # Rotate candidate
        rotated_cand = rotate_img(candidate_img, angle)
            
        h_rot, w_rot = rotated_cand.shape[:2]
        
        for char, tmpl in templates.items():
            h_tmpl, w_tmpl = tmpl.shape[:2]
            if h_tmpl == 0: continue
            
            # Check Height Scale
            # Calculate Observed Scale: Candidate / Template
            observed_scale = h_rot / float(h_tmpl)
            
            if expected_scale is not None:
                # Enforce STRICT scale check (±50% to accommodate small char noise)
                if abs(observed_scale - expected_scale) / expected_scale > 0.50:
                    if debug and angle == 0: 
                         print(f"DEBUG: Rejected scale {observed_scale:.3f} (Exp: {expected_scale}) for {char}. Cand H: {h_rot}, Tmpl H: {h_tmpl}")
                         # Vis rejection
                         if h_rot > 0 and h_tmpl > 0:
                             vis_h = max(h_rot, h_tmpl)
                             vis_w = w_rot + w_tmpl + 10
                             vis = np.full((vis_h, vis_w), 255, dtype=np.uint8)
                             vis[:h_rot, :w_rot] = rotated_cand
                             vis[:h_tmpl, w_rot+10:] = tmpl
                             cv2.imwrite(os.path.join(debug_dir, f"REJECT_scale_{char}_{angle}_{observed_scale:.2f}.png"), vis)
                    continue
            else:
                 if observed_scale < 0.2 or observed_scale > 2.0:
                     if debug and angle == 0: print(f"DEBUG: Rejected loose scale {observed_scale:.3f}")
                     continue

            scale_factor = observed_scale
            if scale_factor == 0: continue
            
            new_w = int(w_tmpl * scale_factor)
            new_h = h_rot 
            
            if new_w <= 0: continue
            
            # --- shape validity check ---
            width_diff = abs(w_rot - new_w)
            max_width = max(w_rot, new_w)
            
            width_tolerance = 0.50 # Relaxed from 0.35
            if width_diff / max_width > width_tolerance:
                if debug and angle == 0: 
                     print(f"DEBUG: Rejected width diff {width_diff} (Max: {max_width}, Tol: {width_tolerance}). Cand W: {w_rot}, Scaled Tmpl W: {new_w}")
                     # Vis rejection
                     if h_rot > 0 and h_tmpl > 0:
                             vis_h = max(h_rot, h_tmpl)
                             vis_w = w_rot + w_tmpl + 10
                             vis = np.full((vis_h, vis_w), 255, dtype=np.uint8)
                             vis[:h_rot, :w_rot] = rotated_cand
                             vis[:h_tmpl, w_rot+10:] = tmpl
                             cv2.imwrite(os.path.join(debug_dir, f"REJECT_width_{char}_{angle}.png"), vis)
                continue
            
            # Check Pixel Mass...
            # Use THRESH_BINARY_INV because input is Black Text on White BG -> We want to count White(Text) pixels
            _, thresh_cand = cv2.threshold(rotated_cand, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, thresh_tmpl = cv2.threshold(tmpl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            cand_mass = cv2.countNonZero(thresh_cand)
            tmpl_mass = cv2.countNonZero(thresh_tmpl)
            
            expected_mass = tmpl_mass * (scale_factor * scale_factor)
            
            if expected_mass > 0:
                mass_ratio = cand_mass / expected_mass
                if mass_ratio < 0.3 or mass_ratio > 2.0:
                    if debug and angle == 0:
                         print(f"DEBUG: Rejected mass ratio {mass_ratio:.2f} (Cand: {cand_mass}, Exp: {expected_mass:.1f})")
                    continue
                    
            # --- Topology / Euler Check ---
            # Distinguish "C" (Open) from "O"/"0"/Pads (Closed)
            # Euler Number = (Number of objects) - (Number of holes)
            # For a single character ROI:
            #   "C", "S", "1", "L" -> 1 object, 0 holes -> Euler = 1
            #   "O", "0", "D", "A", "R", "P", "4", "6", "9" -> 1 object, >=1 holes -> Euler <= 0 (usually 0)
            #   "8", "B" -> 1 object, 2 holes -> Euler = -1
            
            # Simple hole check using contour hierarchy from thresh
            def get_hole_count(binary_img):
                 # Find contours on the binary image (white text)
                 # We need to ensure it's properly padded
                 padded = cv2.copyMakeBorder(binary_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                 contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                 holes = 0
                 if hierarchy is not None:
                     # Hierarchy: [Next, Previous, First_Child, Parent]
                     # If Parent != -1, it's a hole (inner contour)
                     for i in range(len(contours)):
                         if hierarchy[0][i][3] != -1:
                             holes += 1
                 return holes

            tmpl_holes = get_hole_count(thresh_tmpl)
            cand_holes = get_hole_count(thresh_cand)
            
            # Reject if hole count differs excessively
            # "C" (0 holes) vs Pad (1 hole) -> Diff 1 -> Reject
            # "O" (1 hole) vs Pad (1 hole) -> Diff 0 -> Match
            if abs(tmpl_holes - cand_holes) > 0:
                 # Be careful with noise creating small holes.
                 # Filter: If candidate has MORE holes than template (e.g. noise speck inside C), maybe ignore?
                 # But Pad (1) matching C (0) is the main issue.
                 if cand_holes > tmpl_holes:
                      if debug and angle == 0: 
                           print(f"DEBUG: Rejected topology (Holes: Tmpl {tmpl_holes}, Cand {cand_holes})")
                      continue
                      
                 # If candidate has FEWER holes (e.g. 0 broken to C), it might be acceptable if score is high?
                 # But generally topology mismatch is a strong indicator of wrong char.
                 if abs(tmpl_holes - cand_holes) >= 1:
                       if debug and angle == 0: print(f"DEBUG: Rejected topology mismatch {tmpl_holes} vs {cand_holes}")
                       continue
            
            resized_tmpl = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            match_w = max(w_rot, new_w)
            match_h = max(h_rot, new_h)
            
            canvas = np.full((match_h, match_w), 255, dtype=np.uint8)
            y_off = (match_h - h_rot) // 2
            x_off = (match_w - w_rot) // 2
            canvas[y_off:y_off+h_rot, x_off:x_off+w_rot] = rotated_cand
            
            if resized_tmpl.shape[0] > match_h or resized_tmpl.shape[1] > match_w:
                if debug and angle == 0: print("DEBUG: Template larger than canvas")
                continue
                
            res = cv2.matchTemplate(canvas, resized_tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if debug:
                 # Visualize valid attempts regardless of score if it passed pre-checks
                 vis = np.hstack([canvas, cv2.resize(resized_tmpl, (canvas.shape[1], canvas.shape[0]))])
                 fname = f"match_attempt_{char}_{angle}deg_score_{max_val:.2f}.png"
                 cv2.imwrite(os.path.join(debug_dir, fname), vis)
            
            if max_val > best_score:
                best_score = max_val
                best_char = char
                best_angle = angle
            
    return best_char, best_score
