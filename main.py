import cv2
import sys
import os
import argparse
from pathlib import Path

from template_manager import TemplateManager
from image_processor import get_character_candidates, extract_candidate_roi
from matcher import match_character
from grouper import group_characters

def main():
    parser = argparse.ArgumentParser(description="Extract reference designators from PCB images.")
    parser.add_argument("image_path", nargs="?", help="Path to the PCB layout image")
    parser.add_argument("--templates", help="Path to template directory", default="templates")
    parser.add_argument("--font", help="Path to font file for template generation", default=None)
    parser.add_argument("--zoom", help="Zoom factor for PDF rendering (detection)", type=float, default=6.0)
    parser.add_argument("--capture-zoom", help="Zoom factor for High-Res template matching", type=float, default=24.0)
    parser.add_argument("--page", help="Specific page number to process (1-based)", type=int, default=None)
    parser.add_argument("--crop", help="Crop region x,y,w,h (in pixels at --zoom level)", default=None)
    args = parser.parse_args()

    # 0. Interactive Template Selection (if not using font gen)
    # The user asked to choose the template (placement or schematics)
    template_dir = args.templates
    if not args.font:
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            def ask_template_mode_main():
                """Shows a startup dialog to choose the template directory."""
                try:
                    root = tk.Tk()
                    root.title("Select Template Type")
                    
                    # Center the window
                    w, h = 350, 200
                    sw = root.winfo_screenwidth()
                    sh = root.winfo_screenheight()
                    x = (sw - w) // 2
                    y = (sh - h) // 2
                    root.geometry(f"{w}x{h}+{x}+{y}")
                    
                    # Style
                    root.configure(bg="#2d2d30")
                    
                    choice = tk.StringVar(value="")
                    
                    def set_placement():
                        choice.set("placement_templates")
                        root.destroy()
                        
                    def set_schematic():
                        choice.set("schematics_templates")
                        root.destroy()
                        
                    def create_btn(text, cmd):
                        btn = tk.Button(root, text=text, command=cmd, 
                                        font=("Segoe UI", 11), 
                                        bg="#007acc", fg="white", 
                                        activebackground="#005f9e", activeforeground="white",
                                        relief="flat", padx=20, pady=10, cursor="hand2")
                        btn.pack(pady=10, fill="x", padx=40)
                        return btn
                        
                    tk.Label(root, text="Select Template Mode", font=("Segoe UI", 14, "bold"), bg="#2d2d30", fg="white").pack(pady=(20, 10))
                    
                    create_btn("Placement Templates", set_placement)
                    create_btn("Schematics Templates", set_schematic)
                    
                    # Handle window close
                    def on_closing():
                        choice.set("exit")
                        root.destroy()
                        
                    root.protocol("WM_DELETE_WINDOW", on_closing)
                    
                    root.mainloop()
                    return choice.get()
                except Exception as e:
                    print(f"Dialog error: {e}")
                    return "exit"

            selected_mode = ask_template_mode_main()
            if selected_mode == "exit" or not selected_mode:
                print("Exiting...")
                return
            
            template_dir = selected_mode
            print(f"Selected template directory: {template_dir}")

            # 0.1 File Selection if not provided
            if not args.image_path:
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select PDF or Image",
                    filetypes=[("All Files", "*.*"), ("PDF Files", "*.pdf"), ("Images", "*.png;*.jpg;*.jpeg")]
                )
                root.destroy()
                if file_path:
                    args.image_path = file_path
                else:
                    print("No file selected. Exiting.")
                    return

        except ImportError:
            pass
            
    if not args.image_path:
        print("Error: No image_path provided.")
        return

    print(f"Processing {args.image_path}...")

    # 1. Load Templates
    tm = TemplateManager()
    if args.font:
        # Generate on the fly
        import string
        chars = string.ascii_uppercase + string.digits # Add more if needed
        tm.generate_templates_from_font(args.font, chars)
    else:
        # Load from directory selected
        tm.load_templates_from_dir(template_dir)
        
    if not tm.templates:
        print("No templates loaded. Use --font or populate templates directory.")
        return

    # Calculate dynamic candidate size limits based on templates
    max_tmpl_dim = 0
    min_tmpl_dim = 1000
    
    if tm.templates:
         max_tmpl_dim = max(max(t.shape[:2]) for t in tm.templates.values())
         min_tmpl_dim = min(min(t.shape[:2]) for t in tm.templates.values())
    
    # Default loose limits
    cand_max_w = 200
    cand_max_h = 200
    cand_min_w = 5
    cand_min_h = 8
    
    if max_tmpl_dim > 0:
         ratio = 1.0
         # If using directory templates (likely high-res), adjust for zoom ratio
         if not args.font and args.capture_zoom and args.zoom:
             ratio = args.capture_zoom / args.zoom
         
         min_longest_side = 1000
         for t in tm.templates.values():
             min_longest_side = min(min_longest_side, max(t.shape[:2]))
             
         expected_longest = min_longest_side / ratio
         expected_max_size = max_tmpl_dim / ratio
         
         # User filter: 
         limit_max = int(expected_max_size * 2.0)
         limit_strict_min = int(expected_longest * 0.90)
         
         # Safety floor
         limit_max = max(limit_max, 20)
         limit_strict_min = max(limit_strict_min, 4)
         
         cand_max_w = limit_max
         cand_max_h = limit_max
         cand_min_w = 5 # Loose for API
         cand_min_h = 5 
         
         print(f"Dynamic Candidate Size Limits: Max {limit_max}px, Strict Min Longest {limit_strict_min}px")


    # 2. Process Image(s)
    # List of dicts: {'suffix': str, 'img': np.array, 'doc': fitz.Document, 'page': int}
    items_to_process = [] 
    
    if args.image_path.lower().endswith(".pdf"):
        try:
            from pdf_loader import load_pdf, render_page, render_clip
            doc = load_pdf(args.image_path)
            if not doc:
                print("Failed to load PDF.")
                return
            
            print(f"Processing PDF with {len(doc)} pages.")
            
            start_idx = 0
            end_idx = len(doc)
            
            if args.page is not None:
                if 1 <= args.page <= len(doc):
                    start_idx = args.page - 1
                    end_idx = args.page
                    print(f"Processing Page {args.page} only.")
                else:
                    print(f"Error: Page {args.page} is out of range (1-{len(doc)})")
                    return
            else:
                # Interactive Selection
                try:
                    print("Launching interactive page selector...")
                    from pdf_viewer import select_page_from_pdf
                    selected_page = select_page_from_pdf(doc)
                    
                    if selected_page is None:
                        print("Selection cancelled.")
                        return
                        
                    start_idx = selected_page
                    end_idx = selected_page + 1
                    print(f"Selected Page {start_idx + 1}")
                except ImportError:
                    print("Could not import selector. Processing all pages.")
                    pass
                except Exception as e:
                    print("Error in selection (processing all pages instead):", e)
                    pass

            for i in range(start_idx, end_idx):
                print(f"Rendering page {i+1} at zoom {args.zoom} (Detection)...")
                img = render_page(doc, i, zoom=args.zoom)
                if img is None:
                    print(f"Failed to render page {i+1}")
                    continue
                
                # Apply Crop
                if args.crop:
                    try:
                        cx, cy, cw, ch = map(int, args.crop.split(','))
                        # Validate
                        h, w = img.shape[:2]
                        cx = max(0, min(cx, w))
                        cy = max(0, min(cy, h))
                        cw = min(cw, w - cx)
                        ch = min(ch, h - cy)
                        if cw > 0 and ch > 0:
                             print(f"Applying crop to Page {i+1}: {cx},{cy},{cw},{ch}")
                             img = img[cy:cy+ch, cx:cx+cw]
                    except Exception as e:
                        print(f"Crop Error: {e}")

                items_to_process.append({
                    'suffix': f"_page_{i+1}", 
                    'img': img,
                    'doc': doc,
                    'page': i
                })
        except Exception as e:
            print(f"Error loading PDF: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Single image
        img = cv2.imread(args.image_path)
        if img is None:
            print(f"Error loading image {args.image_path}")
            return

        # Handle Crop for items_to_process
        if args.crop:
            try:
                cx, cy, cw, ch = map(int, args.crop.split(','))
                # Validate bounds
                h, w = img.shape[:2]
                cx = max(0, min(cx, w))
                cy = max(0, min(cy, h))
                cw = min(cw, w - cx)
                ch = min(ch, h - cy)
                
                if cw > 0 and ch > 0:
                     print(f"Applying crop: {cx},{cy},{cw},{ch}")
                     img = img[cy:cy+ch, cx:cx+cw]
                items_to_process.append({
                    'suffix': "",
                    'img': img,
                    'doc': None,
                    'page': None
                })
            except Exception as e:
                print(f"Invalid crop format: {e}")
                return
        else:
             items_to_process.append({
                'suffix': "",
                'img': img,
                'doc': None,
                'page': None
            })

    # Process using the refactored system
    for item in items_to_process:
        results = process_page_system(
            base_img=item['img'],
            doc=item['doc'],
            page_num_0based=item['page'],
            tm=tm,
            args=args,
            base_suffix=item['suffix']
        )
        
def process_page_system(base_img, doc, page_num_0based, tm, args, base_suffix="", visualize=True):
    """
    Reusable pipeline for processing a single base image (0-deg).
    Generates rotations, runs detection, grouping, and visualization.
    Returns the list of final filtered labels.
    """
    import numpy as np
    from image_processor import preprocess_from_array
    
    # ... Helper rotation function ...
    def rotate_img_full(image, angle):
        if angle == 0: return image, None
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        b_val = (255, 255, 255)
        if len(image.shape) == 2: b_val = 255
        rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=b_val)
        return rotated, M

    print("Generating rotated tasks (0, 90 CW, 45 CCW)...")
    tasks = []
    
    # 1. 0 degrees
    tasks.append({
        'img': base_img,
        'matrix': None,
        'rotation_name': "0_deg",
        'doc': doc,
        'page': page_num_0based
    })
    
    # 2. 90 degrees Clockwise (-90)
    img_90cw, M_90 = rotate_img_full(base_img, -90)
    tasks.append({
        'img': img_90cw,
        'matrix': M_90,
        'rotation_name': "90_CW",
        'doc': None, # Only use doc for 0 degree high-res crop usually
        'page': None
    })
    
    # 3. 45 degrees Counter-Clockwise (+45)
    img_45ccw, M_45 = rotate_img_full(base_img, 45)
    tasks.append({
        'img': img_45ccw,
        'matrix': M_45,
        'rotation_name': "45_CCW",
        'doc': None,
        'page': None
    })
    
    # Result Aggregation
    all_labels = []
    
    # Calculate limits once
    max_tmpl_dim = 0
    if tm.templates:
         max_tmpl_dim = max(max(t.shape[:2]) for t in tm.templates.values())
         
    # Setup candidate params
    # Default loose limits
    cand_max_w = 200
    cand_max_h = 200
    cand_min_w = 5
    cand_min_h = 8
    limit_strict_min = 5
    
    if max_tmpl_dim > 0:
         ratio = 1.0
         if not args.font and args.capture_zoom and args.zoom:
             ratio = args.capture_zoom / args.zoom
         
         min_longest_side = 1000
         for t in tm.templates.values():
             min_longest_side = min(min_longest_side, max(t.shape[:2]))
             
         expected_longest = min_longest_side / ratio
         expected_max_size = max_tmpl_dim / ratio
         
         limit_max = int(expected_max_size * 2.0)
         limit_strict_min = int(expected_longest * 0.90)
         
         limit_max = max(limit_max, 20)
         limit_strict_min = max(limit_strict_min, 4)
         
         cand_max_w = limit_max
         cand_max_h = limit_max
    
    # PROCESSING LOOP
    for task in tasks:
        img = task['img']
        rot_name = task['rotation_name']
        matrix = task['matrix']
        t_doc = task['doc']
        t_page = task['page']
        
        print(f"--- Processing {rot_name} ---")
        
        try:
            line_len = 100
            if max_tmpl_dim > 0:
                 ratio = 1.0
                 if not args.font and args.capture_zoom and args.zoom:
                     ratio = args.capture_zoom / args.zoom
                 curr_expected_max_size = max_tmpl_dim / ratio
                 line_len = int(curr_expected_max_size * 2.0)
                 line_len = max(line_len, 50)
            elif doc is not None:
                 line_len = max(40, int(args.zoom * 10))

            _, gray_img, binary_img = preprocess_from_array(img, min_line_length=line_len)
            
            # Candidates
            raw_candidates = get_character_candidates(binary_img, min_w=cand_min_w, min_h=cand_min_h, max_w=cand_max_w, max_h=cand_max_h)
            
            candidates = []
            if max_tmpl_dim > 0:
                for c in raw_candidates:
                    longest = max(c['w'], c['h'])
                    ratio = c['h'] / float(c['w']) if c['w'] > 0 else 0
                    if longest >= limit_strict_min:
                        candidates.append(c)
                    elif longest >= limit_strict_min * 0.5 and ratio > 2.0:
                        candidates.append(c)
            else:
                candidates = raw_candidates
                
            print(f"Found {len(candidates)} candidates.")
            
            # High Res Setup
            high_res_img_task = None
            zoom_ratio = 1.0
            
            if t_doc is not None and t_page is not None:
                try:
                    high_res_img_task = render_page(t_doc, t_page, zoom=args.capture_zoom)
                    
                    # Apply Crop if args.crop exists
                    if high_res_img_task is not None and args.crop:
                         cx, cy, cw, ch = map(int, args.crop.split(','))
                         ratio_crop = args.capture_zoom / args.zoom
                         hcx, hcy = int(cx * ratio_crop), int(cy * ratio_crop)
                         hcw, hch = int(cw * ratio_crop), int(ch * ratio_crop)
                         hh, hw = high_res_img_task.shape[:2]
                         hcx = max(0, min(hcx, hw))
                         hcy = max(0, min(hcy, hh))
                         hcw = min(hcw, hw - hcx)
                         hch = min(hch, hh - hcy)
                         if hcw > 0 and hch > 0:
                             high_res_img_task = high_res_img_task[hcy:hcy+hch, hcx:hcx+hcw]

                    if high_res_img_task is not None:
                        high_res_img_task = cv2.cvtColor(high_res_img_task, cv2.COLOR_BGR2GRAY)
                        _, high_res_img_task = cv2.threshold(high_res_img_task, 180, 255, cv2.THRESH_BINARY)
                        zoom_ratio = args.capture_zoom / args.zoom
                except:
                    pass
            
            # Match
            matches = []
            for idx, cand in enumerate(candidates):
                roi = None
                if high_res_img_task is not None:
                    try:
                        x, y, w, h = cand['bbox']
                        x_hi, y_hi = int(x * zoom_ratio), int(y * zoom_ratio)
                        w_hi, h_hi = int(w * zoom_ratio), int(h * zoom_ratio)
                        h_img, w_img = high_res_img_task.shape
                        pad = 2
                        x_hi = max(0, x_hi - pad)
                        y_hi = max(0, y_hi - pad)
                        w_hi = min(w_img - x_hi, w_hi + 2*pad)
                        h_hi = min(h_img - y_hi, h_hi + 2*pad)
                        if w_hi > 0 and h_hi > 0:
                            roi = high_res_img_task[y_hi:y_hi+h_hi, x_hi:x_hi+w_hi]
                    except: pass
                
                if roi is None:
                     roi = extract_candidate_roi(gray_img, cand)
                
                if roi is None or roi.size == 0: continue
                
                char, score, angle = match_character(roi, tm.templates, rotation_angles=[0])
                if score > 0.5:
                    match_data = cand.copy()
                    match_data['char'] = char
                    match_data['score'] = score
                    match_data['angle'] = angle
                    matches.append(match_data)
            
            task_labels = group_characters(matches)
            print(f"Detected {len(task_labels)} labels in {rot_name}.")
            
            # Transform Back
            for l in task_labels:
                x, y, w, h = l['bbox']
                pts = np.array([[x, y],[x + w, y],[x + w, y + h],[x, y + h]], dtype=np.float32)
                
                if matrix is not None:
                    inv_matrix = cv2.invertAffineTransform(matrix)
                    original_pts = cv2.transform(np.array([pts]), inv_matrix)[0]
                else:
                    original_pts = pts
                
                poly_points = original_pts.astype(int).tolist()
                center_x = int(np.mean(original_pts[:, 0]))
                center_y = int(np.mean(original_pts[:, 1]))
                
                all_labels.append({
                    'text': l['text'],
                    'type': l['type'],
                    'score': l['score'],
                    'rotation_found': rot_name,
                    'poly': poly_points,
                    'center': (center_x, center_y),
                    'bbox_local': (x,y,w,h)
                })

        except Exception as e:
            print(f"Error in {rot_name}: {e}")
            import traceback
            traceback.print_exc()

    # NMS Filtering on accumulated labels
    def get_poly_bbox(poly_pts):
        arr = np.array(poly_pts)
        x = np.min(arr[:,0])
        y = np.min(arr[:,1])
        w = np.max(arr[:,0]) - x
        h = np.max(arr[:,1]) - y
        return (x, y, w, h)

    def compute_iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def filter_overlapping(labels, threshold=0.2):
        def priority(l): return (1000 if l['type'] == 'COMPONENT' else 0) + l['score']
        sorted_indices = sorted(range(len(labels)), key=lambda i: priority(labels[i]), reverse=True)
        keep = []
        for i in sorted_indices:
            current_poly = labels[i]['poly']
            current_box = get_poly_bbox(current_poly)
            is_overlap = False
            for k_idx in keep:
                kept_box = get_poly_bbox(labels[k_idx]['poly'])
                if compute_iou(current_box, kept_box) > threshold:
                    is_overlap = True; break
            if not is_overlap: keep.append(i)
        return [labels[i] for i in keep]

    final_labels = filter_overlapping(all_labels)
    
    # VISUALIZATION
    if visualize:
        base_vis = base_img.copy()
        for l in final_labels:
            poly = np.array(l['poly'], dtype=np.int32)
            color = (0, 255, 0) if l['type'] == 'COMPONENT' else (0, 165, 255)
            thick = 2 if l['type'] == 'COMPONENT' else 1
            cv2.polylines(base_vis, [poly], isClosed=True, color=color, thickness=thick)
            
            # Calculate optimal text position
            txt_pos = tuple(l['poly'][0])    
            cv2.putText(base_vis, l['text'], txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4) # outline
            cv2.putText(base_vis, l['text'], txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        title = f"Combined Results {base_suffix}"
        cv2.imshow(title, base_vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    
    return final_labels

if __name__ == "__main__":
    main()
