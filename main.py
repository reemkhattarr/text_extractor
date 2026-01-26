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
        items_to_process.append({
            'suffix': "",
            'img': img,
            'doc': None,
            'page': None
        })

    # PROPOSED CHANGE: Generate rotated dictionary tasks
    import numpy as np
    
    def rotate_img_full(image, angle):
        if angle == 0: return image, None
        
        # General rotation
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        # Fill with white (background)
        b_val = (255, 255, 255)
        if len(image.shape) == 2: # Grayscale
            b_val = 255
            
        rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=b_val)
        return rotated, M

    print("Generating rotated tasks (0°, 90° CW, 45° CCW)...")
    final_tasks = []
    
    # Store aggregated results: page_id -> { 'orig_img': img, 'labels': [] }
    # page_id can be the page index or suffix
    
    for item in items_to_process:
        base_img = item['img']
        base_suffix = item['suffix']
        page_id = item['page'] if item['page'] is not None else 0
        
        # 1. 0 degrees (Original)
        item['matrix'] = None
        item['rotation_name'] = "0_deg"
        item['page_id'] = page_id
        final_tasks.append(item)
        
        # 2. 90 degrees Clockwise (-90)
        img_90cw, M_90 = rotate_img_full(base_img, -90)
        final_tasks.append({
            'suffix': base_suffix + "_rot90CW",
            'img': img_90cw,
            'doc': None,
            'page': None,
            'matrix': M_90,
            'rotation_name': "90_CW",
            'page_id': page_id
        })
        
        # 3. 45 degrees Counter-Clockwise (+45)
        img_45ccw, M_45 = rotate_img_full(base_img, 45)
        final_tasks.append({
            'suffix': base_suffix + "_rot45CCW",
            'img': img_45ccw,
            'doc': None,
            'page': None,
            'matrix': M_45,
            'rotation_name': "45_CCW",
            'page_id': page_id
        })
        
    items_to_process = final_tasks
    
    # Dictionary to hold final results per original page
    # key: page_id
    # value: { 'orig_img': numpy_array, 'labels': list_of_dicts, 'filepath': str }
    page_results_map = {}

    from image_processor import preprocess_from_array

    for item in items_to_process:
        suffix = item['suffix']
        orig_img = item['img']
        doc = item['doc']
        page_num = item['page']
        matrix = item.get('matrix')
        page_id = item.get('page_id')
        rot_name = item.get('rotation_name', '0')

        # Initialize result entry for this page if strictly original (0 deg) info needed
        if page_id not in page_results_map:
             page_results_map[page_id] = { 'labels': []}

        # If this is the 0-degree image, save it as the base for visualization
        if matrix is None:
             page_results_map[page_id]['orig_img'] = orig_img
             page_results_map[page_id]['base_filepath'] = args.image_path

        print(f"--- Processing {suffix} ({rot_name}) ---")
        try:
            # Determine line removal threshold based on template size
            # User request: "remove only lines ... at least 2x longer ... than the biggest template size"
            line_len = 100 # Fallback default
            
            # We need to access expected_max_size. It was calculated in the setup phase if max_tmpl_dim > 0.
            # To be safe and avoid scope issues, we can re-derive the logic or check locals.
            # But better: Check if we have templates and calculate dynamically.
            
            if max_tmpl_dim > 0:
                 # Re-calculate ratio valid for this run
                 ratio = 1.0
                 if not args.font and args.capture_zoom and args.zoom:
                     ratio = args.capture_zoom / args.zoom
                     
                 curr_expected_max_size = max_tmpl_dim / ratio
                 line_len = int(curr_expected_max_size * 2.0)
                 
                 # Ensure it's not too small (e.g. if templates are tiny)
                 line_len = max(line_len, 50)
                 print(f"DEBUG: Using line_len={line_len} (2 * MaxTmpl {int(curr_expected_max_size)})")
            elif doc is not None:
                 line_len = max(40, int(args.zoom * 10))

            
            _, gray_img, binary_img = preprocess_from_array(orig_img, min_line_length=line_len)
            
            # DEBUG: Save binary image
            cv2.imwrite(f"debug_binary_{suffix}.png", binary_img)
            print(f"Saved debug_binary_{suffix}.png")
        except Exception as e:
            print(f"Error processing image section {suffix}: {e}")
            continue

        # 3. Get Candidates
        raw_candidates = get_character_candidates(binary_img, min_w=cand_min_w, min_h=cand_min_h, max_w=cand_max_w, max_h=cand_max_h)
        # Apply Strict Filter
        candidates = []
        if max_tmpl_dim > 0:
            for c in raw_candidates:
                longest = max(c['w'], c['h'])
                ratio = c['h'] / float(c['w']) if c['w'] > 0 else 0
                
                # 1. Standard Size Check
                if longest >= limit_strict_min:
                    candidates.append(c)
                # 2. Rescue Small Thin Characters (e.g. '1', 'l', 'I')
                # These might be smaller than the average letter but are valid if they are tall/thin.
                # 'D' noise is usually blocky (ratio ~1.0-1.5), so checking ratio > 2.0 filters it out.
                elif longest >= limit_strict_min * 0.5 and ratio > 2.0:
                    candidates.append(c)
        else:
            candidates = raw_candidates
            
        print(f"Found {len(candidates)} candidates.")
        
        # Optimize PDF High-Res Extraction (Only for 0 deg / original page)
        high_res_img = None
        zoom_ratio = 1.0
        
        if doc is not None and page_num is not None:
            try:
                # Render logic...
                high_res_img = render_page(doc, page_num, zoom=args.capture_zoom)
                if high_res_img is not None:
                    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2GRAY)
                    _, high_res_img = cv2.threshold(high_res_img, 180, 255, cv2.THRESH_BINARY)
                    zoom_ratio = args.capture_zoom / args.zoom
            except Exception as e:
                pass
        
        # 4. Match Candidates
        matches = []
        import time
        start_time = time.time()
        
        for idx, cand in enumerate(candidates):
            roi = None
            if high_res_img is not None:
                # Fast Crop Strategy for High Res
                try:
                    x, y, w, h = cand['bbox']
                    x_hi = int(x * zoom_ratio)
                    y_hi = int(y * zoom_ratio)
                    w_hi = int(w * zoom_ratio)
                    h_hi = int(h * zoom_ratio)
                    h_img, w_img = high_res_img.shape
                    x_hi = max(0, min(x_hi, w_img - 1))
                    y_hi = max(0, min(y_hi, h_img - 1))
                    pad = 2
                    x_hi = max(0, x_hi - pad)
                    y_hi = max(0, y_hi - pad)
                    w_hi = min(w_img - x_hi, w_hi + 2*pad)
                    h_hi = min(h_img - y_hi, h_hi + 2*pad)
                    
                    if w_hi > 0 and h_hi > 0:
                        roi = high_res_img[y_hi:y_hi+h_hi, x_hi:x_hi+w_hi]
                except:
                    pass
            
            if roi is None:
               roi = extract_candidate_roi(gray_img, cand)
            
            if roi is None or roi.size == 0:
                continue

            char, score, angle = match_character(roi, tm.templates, rotation_angles=[0])
            
            if score > 0.5:
                # print(f"Match: {char} ({score:.2f}) at {cand['x']},{cand['y']}")
                pass
            elif score > 0.3:
                 print(f"Low Score Match: {char} ({score:.2f}) at {cand['x']},{cand['y']} Size: {cand['w']}x{cand['h']}")
            
            if score > 0.5:
                match_data = cand.copy()
                match_data['char'] = char
                match_data['score'] = score
                match_data['angle'] = angle
                matches.append(match_data)
        
        labels = group_characters(matches)
        
        # Filter (keep all but tag them) or keep only COMPONENT?
        # User said "Show all extracted labels" usually means components + maybe unknowns? 
        # But consistent with previous step, we likely care about components.
        # Let's keep all valid detected labels for potential review
        
        print(f"Detected {len(labels)} labels in {rot_name} pass.")
        
        # TRANSFORM BACK TO ORIGINAL COORDINATES
        for l in labels:
            x, y, w, h = l['bbox']
            # Create the 4 corners of the bbox
            # (x, y) is top-left
            pts = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            # Add extra dim for matrix mult
            # pts array shape: (4, 2)
            # We need to reshape to (4, 1, 2) for perspectiveTransform or just use manual math
            
            if matrix is not None:
                # Invert the affine matrix
                # matrix is 2x3. 
                # OpenCV invertAffineTransform expects 2x3
                inv_matrix = cv2.invertAffineTransform(matrix)
                # transform points
                original_pts = cv2.transform(np.array([pts]), inv_matrix)[0]
            else:
                original_pts = pts
            
            # Store data
            # Convert to list of tuples for easier handling
            poly_points = original_pts.astype(int).tolist()
             
            # Calculate a representative center for text listing
            center_x = int(np.mean(original_pts[:, 0]))
            center_y = int(np.mean(original_pts[:, 1]))
            
            page_results_map[page_id]['labels'].append({
                'text': l['text'],
                'type': l['type'],
                'score': l['score'],
                'rotation_found': rot_name,
                'poly': poly_points,
                'center': (center_x, center_y),
                'bbox_local': (x,y,w,h) # debug info
            })

    # --- FINAL VISUALIZATION & SAVING ---
    for p_id, res in page_results_map.items():
        if 'orig_img' not in res:
            print(f"Skipping page {p_id} (No base image found)")
            continue
            
        base_vis = res['orig_img'].copy()
        all_labels = res['labels']
        base_path = res.get('base_filepath', 'output')
        
        print(f"Aggregation: Page {p_id} has {len(all_labels)} total detected labels.")
        
        # Prepare Text File
        txt_path = f"{Path(base_path).stem}_page_{p_id+1}_COMBINED.txt"
        with open(txt_path, "w") as f:
            f.write(f"Combined Extraction Report for Page {p_id+1}\n")
            f.write(f"Source: {base_path}\n")
            f.write("Includes 0deg, 90deg CW, 45deg CCW passes.\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Label':<15} | {'Type':<5} | {'Conf':<5} | {'Found @':<8} | {'Center (x,y)'}\n")
            f.write("-" * 80 + "\n")
            
            # Draw
            
            # --- NMS / Overlap Filtering ---
            def get_poly_bbox(poly_pts):
                arr = np.array(poly_pts)
                x = np.min(arr[:,0])
                y = np.min(arr[:,1])
                w = np.max(arr[:,0]) - x
                h = np.max(arr[:,1]) - y
                return (x, y, w, h)

            def compute_iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
                yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = boxA[2] * boxA[3]
                boxBArea = boxB[2] * boxB[3]
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                return iou

            def filter_overlapping(labels, threshold=0.3):
                # Sort by Priority: COMPONENT first, then Score
                # We want to keep the best ones, so we sort descending
                def priority(l):
                    type_score = 1000 if l['type'] == 'COMPONENT' else 0
                    return type_score + l['score']
                
                sorted_indices = sorted(range(len(labels)), key=lambda i: priority(labels[i]), reverse=True)
                keep = []
                
                for i in sorted_indices:
                    current_poly = labels[i]['poly']
                    current_box = get_poly_bbox(current_poly)
                    
                    is_overlap = False
                    for k_idx in keep:
                        kept_label = labels[k_idx]
                        kept_box = get_poly_bbox(kept_label['poly'])
                        
                        if compute_iou(current_box, kept_box) > threshold:
                            is_overlap = True
                            break
                            
                    if not is_overlap:
                        keep.append(i)
                        
                return [labels[i] for i in keep]

            filtered_labels = filter_overlapping(all_labels, threshold=0.2)
            print(f"NMS applied: Reduced {len(all_labels)} -> {len(filtered_labels)} labels.")
            all_labels = filtered_labels
            
            for l in all_labels:
                poly = np.array(l['poly'], dtype=np.int32)
                
                # Color Coding
                if l['type'] == 'COMPONENT':
                    color = (0, 255, 0) # Green
                    thick = 2
                else:
                    color = (0, 165, 255) # Orange for unknown
                    thick = 1
                
                # Draw Polygon
                cv2.polylines(base_vis, [poly], isClosed=True, color=color, thickness=thick)
                
                # Put Text at the top-left-ish corner (first point) or center
                txt_pos = tuple(l['poly'][0])    
                # Ensure text is on screen?
                
                cv2.putText(base_vis, l['text'], txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4) # outline
                cv2.putText(base_vis, l['text'], txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Write to file
                f.write(f"{l['text']:<15} | {('CP' if l['type']=='COMPONENT' else '?'):<5} | {l['score']:.2f} | {l['rotation_found']:<8} | {l['center']}\n")
                
        # Save Image
        out_img_path = f"{Path(base_path).stem}_page_{p_id+1}_COMBINED.png"
        cv2.imwrite(out_img_path, base_vis)
        print(f"Saved Combined Text: {txt_path}")
        print(f"Saved Combined Image: {out_img_path}")
        
        # Show
        max_h = 900
        h_vis, w_vis = base_vis.shape[:2]
        if h_vis > max_h:
             scale = max_h / h_vis
             vis_disp = cv2.resize(base_vis, None, fx=scale, fy=scale)
        else:
             vis_disp = base_vis
             
        title = f"Combined Results Page {p_id+1}"
        cv2.imshow(title, vis_disp)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyWindow(title)


if __name__ == "__main__":
    main()
