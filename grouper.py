import numpy as np
import re

def group_characters(matched_chars, adaptive_thresh=True):
    """
    Groups individual characters into labels.
    Supports Horizontal and Vertical orientation (labels are either H or V).
    
    Args:
        matched_chars: List of dicts {'char', 'x', 'y', 'w', 'h', 'score'}
    
    Returns:
        List of dicts {'text', 'bbox', 'score', 'type', 'orientation'}
    """
    # 1. Deduplicate/Filter
    # Remove overlapping single-character detections (e.g. jittered duplicates)
    sorted_chars = sorted(matched_chars, key=lambda x: x['score'], reverse=True)
    valid_chars = []
    
    def get_char_box(c):
        return (c['x'], c['y'], c['w'], c['h'])

    def compute_char_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    for c in sorted_chars:
        if c['score'] < 0.5: continue
        
        is_duplicate = False
        c_box = get_char_box(c)
        for existing in valid_chars:
            if compute_char_iou(c_box, get_char_box(existing)) > 0.4: # Strict overlap check
                is_duplicate = True
                break
        if not is_duplicate:
            valid_chars.append(c)

    if not valid_chars:
        return []

    # Assign IDs
    for i, c in enumerate(valid_chars):
        c['_id'] = i
        c['_used'] = False

    labels = []

    # --- Pass 1: Horizontal Grouping ---
    # Build Adjacency Dictionary for Horizontal Links
    h_adj = {i: [] for i in range(len(valid_chars))}
    
    sorted_indices = sorted(range(len(valid_chars)), key=lambda k: valid_chars[k]['x'])
    
    for idx_i in range(len(sorted_indices)):
        i = sorted_indices[idx_i]
        c1 = valid_chars[i]
        
        # Look ahead
        for idx_j in range(idx_i + 1, len(sorted_indices)):
            j = sorted_indices[idx_j]
            c2 = valid_chars[j]
            
            # X-Gap check
            gap = c2['x'] - (c1['x'] + c1['w'])
            
            # Dynamic Threshold
            # Fix for narrow characters (like '1'): Use height as a fallback proxy for font size
            avg_w = (c1['w'] + c2['w']) / 2
            avg_h = (c1['h'] + c2['h']) / 2
            
            # Allow gap up to 1.5x width OR 0.5x height (whichever is larger)
            # This helps narrow chars which might have wide spacing relative to their own width
            gap_limit = max(avg_w * 1.5, avg_h * 0.6) if adaptive_thresh else 30
            
            if gap > gap_limit:
                # Break early optimization
                if gap > gap_limit * 2.0: 
                    break
                continue
                
            if _is_h_linked(c1, c2, gap_limit):
                h_adj[i].append(j)
                h_adj[j].append(i)

    # Find H Components
    h_components = _find_components(h_adj)
    
    for comp in h_components:
        if len(comp) > 1:
            # Check consistency? (e.g. are they strictly linear or a blob?)
            # Valid H-Label
            comp_indices = list(comp)
            comp_chars = [valid_chars[i] for i in comp_indices]
            comp_chars.sort(key=lambda c: c['x']) # L->R
            
            label = _form_label(comp_chars, "H")
            labels.append(label)
            
            # Mark Used
            for i in comp_indices:
                valid_chars[i]['_used'] = True

    # --- Pass 2: Vertical Grouping ---
    # Only consider unused characters
    unused_indices = [i for i, c in enumerate(valid_chars) if not c['_used']]
    
    v_adj = {i: [] for i in unused_indices}
    
    # Sort by Y
    sorted_indices_v = sorted(unused_indices, key=lambda k: valid_chars[k]['y'])
    
    for idx_i in range(len(sorted_indices_v)):
        i = sorted_indices_v[idx_i]
        c1 = valid_chars[i]
        
        for idx_j in range(idx_i + 1, len(sorted_indices_v)):
            j = sorted_indices_v[idx_j]
            c2 = valid_chars[j]
            
            gap = c2['y'] - (c1['y'] + c1['h'])
            avg_h = (c1['h'] + c2['h']) / 2
            gap_limit = avg_h * 1.0 if adaptive_thresh else 20
            
            if gap > gap_limit * 2:
                break
                
            if _is_v_linked(c1, c2, gap_limit):
                v_adj[i].append(j)
                v_adj[j].append(i)
                
    v_components = _find_components(v_adj)
    
    for comp in v_components:
        if len(comp) > 1:
            comp_indices = list(comp)
            comp_chars.sort(key=lambda c: c['y']) # Default Top->Bottom
            
            # --- Check Vertical Text Direction ---
            # If text is rotated 270 degrees (90 CW), it usually reads Bottom -> Top.
            # If text is rotated 90 degrees (90 CCW), it usually reads Top -> Bottom (like standard Vertical Japanese/Chinese, though rare in PCB).
            # If upright (0), it reads Top -> Bottom.
            
            angles = [c.get('angle', 0) for c in comp_chars]
            if angles:
                # Simple mode
                from collections import Counter
                most_common_angle = Counter(angles).most_common(1)[0][0]
                
                # Logic: If angle is 270 (which corresponds to 90 CW rotation of the TEMPLATE), 
                # effectively the text is running "Up" the page.
                if most_common_angle == 270:
                    comp_chars.reverse() # Reverse Y-sort to get Bottom -> Top
            
            label = _form_label(comp_chars, "V")
            labels.append(label)
            
            for i in comp_indices:
                valid_chars[i]['_used'] = True

    # --- Pass 3: Singletons ---
    # Maybe a singleton is a label (e.g. "R"?)
    # User said "one or two letters and then one or two digits".
    # So "R" implies incomplete. "C1" is complete.
    # We still return them, but mark type.
    for i, c in enumerate(valid_chars):
        if not c['_used']:
            labels.append(_form_label([c], "SINGLE"))
            
    # --- Classify Labels ---
    for l in labels:
        l['type'] = _classify_label(l['text'])
        
    # Sort output (e.g. Top-Left to Bottom-Right)
    labels.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
        
    return labels

def _find_components(adj_list):
    """Simple BFS/DFS to find connected components."""
    visited = set()
    components = []
    
    nodes = list(adj_list.keys())
    
    for node in nodes:
        if node not in visited:
            stack = [node]
            visited.add(node)
            comp = set([node])
            
            while stack:
                curr = stack.pop()
                for neighbor in adj_list.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        comp.add(neighbor)
                        stack.append(neighbor)
            components.append(comp)
    return components

def _is_h_linked(c1, c2, gap_limit):
    # Constraint: If characters are rotated 90/270, they (usually) form Vertical stacks.
    # Therefore, we should NOT group them horizontally.
    a1 = c1.get('angle', 0)
    a2 = c2.get('angle', 0)
    
    if (a1 in [90, 270]) or (a2 in [90, 270]):
        # Strict: If EITHER is vertical, don't H-link? 
        # Or strictly if BOTH are vertical? 
        # Usually checking one is enough to break a mixed group, but checking both is safer.
        # Let's say if BOTH are vertical orientation, they should definitely not form a H-line.
        if (a1 in [90, 270]) and (a2 in [90, 270]):
             return False
        
    # Alignment: Y Overlap
    y_inter_low = max(c1['y'], c2['y'])
    y_inter_high = min(c1['y'] + c1['h'], c2['y'] + c2['h'])
    y_overlap = max(0, y_inter_high - y_inter_low)
    
    min_h = min(c1['h'], c2['h'])
    # Must share at least 50% of the smaller height
    if y_overlap < 0.4 * min_h: 
        return False
        
    # Distance
    # Assume sorted by X? No, c1/c2 passed arbitrarily in loop but sorted in outer.
    # But let's be safe.
    left = c1 if c1['x'] < c2['x'] else c2
    right = c2 if left is c1 else c1
    
    dist = right['x'] - (left['x'] + left['w'])
    
    # Allow slight overlap (negative dist) up to 30% width (kerning?)
    # But usually PCB text is monospaced or separated.
    min_w = min(c1['w'], c2['w'])
    if dist < -0.3 * min_w: 
        return False # Too much overlap, maybe nested
        
    if dist > gap_limit:
        return False
        
    return True

def _is_v_linked(c1, c2, gap_limit):
    # Alignment: X Overlap
    x_inter_low = max(c1['x'], c2['x'])
    x_inter_high = min(c1['x'] + c1['w'], c2['x'] + c2['w'])
    x_overlap = max(0, x_inter_high - x_inter_low)
    
    min_w = min(c1['w'], c2['w'])
    if x_overlap < 0.4 * min_w:
        return False
        
    # Distance
    top = c1 if c1['y'] < c2['y'] else c2
    bot = c2 if top is c1 else c1
    
    dist = bot['y'] - (top['y'] + top['h'])
    
    # Adaptive Gap Limit Override for Rotated Text
    # If characters are rotated 90 or 270, their "Height" in bbox (h) is actually the character width.
    # The "font size" corresponds to bbox Width (w).
    # We should use Width as the basis for the vertical gap threshold.
    a1 = c1.get('angle', 0)
    a2 = c2.get('angle', 0)
    
    # Check if vertically oriented text (90 or 270)
    if (a1 in [90, 270]) and (a2 in [90, 270]):
        avg_dim = (c1['w'] + c2['w']) / 2
        # Use a generous gap limit relative to font size (1.0x font size is usually safe)
        adaptive_limit = avg_dim * 1.2 
        # Override the passed gap_limit (which was based on h)
        gap_limit = max(gap_limit, adaptive_limit)
    
    min_h = min(c1['h'], c2['h'])
    if dist < -0.3 * min_h:
        return False
        
    if dist > gap_limit:
        return False
        
    return True

def _form_label(chars, orientation):
    text = "".join([c['char'] for c in chars])
    
    min_x = min(c['x'] for c in chars)
    min_y = min(c['y'] for c in chars)
    max_x = max(c['x'] + c['w'] for c in chars)
    max_y = max(c['y'] + c['h'] for c in chars)
    
    avg_score = sum(c['score'] for c in chars) / len(chars)
    
    return {
        'text': text,
        'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
        'score': avg_score,
        'orientation': orientation,
        'chars': chars
    }

def _classify_label(text):
    # Regex for Component Label: 
    # 1-2 Letters + 1-2 Digits (user said 1-2, but lets support 1-3)
    # e.g. R1, C12, SW1, U100
    
    if re.match(r'^[A-Z]{1,2}[0-9]{1,3}$', text):
        return "COMPONENT"
    
    if re.match(r'^[A-Z]+$', text):
        return "TEXT_ONLY"
        
    if re.match(r'^[0-9]+$', text):
        return "NUM_ONLY"
        
    return "UNKNOWN"
