import numpy as np
import networkx as nx

def group_characters(matched_chars, x_gap_thresh=25, y_gap_thresh=15, score_thresh=0.6):
    """
    Groups individual characters into labels using a priority-based logic:
    1. Identify Horizontal Chains (length > 1).
    2. Identify Vertical Chains from remaining characters (length > 1).
    3. Remaining characters are Singletons.
    """
    # 1. Filter by score
    valid_chars = [c for c in matched_chars if c['score'] >= score_thresh]
    if not valid_chars:
        return []

    # Map index -> char for easy access
    char_map = {i: c for i, c in enumerate(valid_chars)}
    remaining_indices = set(char_map.keys())
    
    final_labels = []

    # --- Pass 1: Horizontal Grouping ---
    G_h = nx.Graph()
    for i in remaining_indices:
        G_h.add_node(i)
        
    # Add H-Edges
    # Optimization: Sort by y to reduce N^2 checks? 
    # N is small enough (<500 usually)
    idx_list = list(remaining_indices)
    for idx_i in range(len(idx_list)):
        i = idx_list[idx_i]
        c1 = char_map[i]
        for idx_j in range(idx_i + 1, len(idx_list)):
            j = idx_list[idx_j]
            c2 = char_map[j]
            
            if _is_h_linked(c1, c2, x_gap_thresh):
                G_h.add_edge(i, j)
                
    # Extract Non-Trivial H Components
    h_components = list(nx.connected_components(G_h))
    for comp in h_components:
        if len(comp) > 1:
            # Form Label
            comp_indices = list(comp)
            comp_chars = [char_map[i] for i in comp_indices]
            comp_chars.sort(key=lambda c: c['x']) # Sort L->R
            final_labels.append(_form_label(comp_chars))
            
            # Remove from remaining
            for i in comp_indices:
                if i in remaining_indices:
                    remaining_indices.remove(i)
                    
    # --- Pass 2: Vertical Grouping ---
    if remaining_indices:
        G_v = nx.Graph()
        for i in remaining_indices:
            G_v.add_node(i)
            
        idx_list = list(remaining_indices)
        for idx_i in range(len(idx_list)):
            i = idx_list[idx_i]
            c1 = char_map[i]
            for idx_j in range(idx_i + 1, len(idx_list)):
                j = idx_list[idx_j]
                c2 = char_map[j]
                
                if _is_v_linked(c1, c2, y_gap_thresh):
                    G_v.add_edge(i, j)
                    
        v_components = list(nx.connected_components(G_v))
        for comp in v_components:
            if len(comp) > 1:
                # Vertical Label
                comp_indices = list(comp)
                comp_chars = [char_map[i] for i in comp_indices]
                comp_chars.sort(key=lambda c: c['y']) # Sort Top->Bottom
                final_labels.append(_form_label(comp_chars))
                
                for i in comp_indices:
                    remaining_indices.remove(i)
                    
    # --- Pass 3: Singletons ---
    for i in remaining_indices:
        final_labels.append(_form_label([char_map[i]]))
        
    return final_labels

def _is_h_linked(c1, c2, thresh):
    # 1. Alignment Check: Must share significant Y-range
    y_overlap = min(c1['y']+c1['h'], c2['y']+c2['h']) - max(c1['y'], c2['y'])
    min_h = min(c1['h'], c2['h'])
    
    if y_overlap < 0.5 * min_h: 
        return False
        
    # 2. Proximity Check
    if c1['x'] < c2['x']:
        dist = c2['x'] - (c1['x'] + c1['w'])
    else:
        dist = c1['x'] - (c2['x'] + c2['w'])
        
    # Allow small negative distance (overlap) but not too much (nested?)
    # Nested check: if one is inside another, it's not a sequence? 
    # For now, simplistic distance check
    return -10 < dist < thresh

def _is_v_linked(c1, c2, thresh):
    # 1. Alignment Check: Must share significant X-range
    x_overlap = min(c1['x']+c1['w'], c2['x']+c2['w']) - max(c1['x'], c2['x'])
    min_w = min(c1['w'], c2['w'])
    
    if x_overlap < 0.5 * min_w:
        return False
        
    # 2. Proximity Check
    if c1['y'] < c2['y']:
        dist = c2['y'] - (c1['y'] + c1['h'])
    else:
        dist = c1['y'] - (c2['y'] + c2['h'])
        
    return -10 < dist < thresh

def _form_label(chars):
    """Combines characters into a label object."""
    text = "".join([c['char'] for c in chars])
    
    min_x = min(c['x'] for c in chars)
    min_y = min(c['y'] for c in chars)
    max_x = max(c['x'] + c['w'] for c in chars)
    max_y = max(c['y'] + c['h'] for c in chars)
    
    avg_score = sum(c['score'] for c in chars) / len(chars)
    
    return {
        'text': text,
        'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
        'score': avg_score
    }
