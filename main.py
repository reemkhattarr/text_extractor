import cv2
import sys
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

    from image_processor import preprocess_from_array

    for item in items_to_process:
        suffix = item['suffix']
        orig_img = item['img']
        doc = item['doc']
        page_num = item['page']

        print(f"--- Processing{suffix} ---")
        try:
            _, gray_img, binary_img = preprocess_from_array(orig_img)
        except Exception as e:
            print(f"Error processing image section{suffix}: {e}")
            continue

        # 3. Get Candidates
        candidates = get_character_candidates(binary_img)
        print(f"Found {len(candidates)} candidate regions.")
        
        # Optimize PDF High-Res Extraction
        high_res_img = None
        zoom_ratio = 1.0
        
        if doc is not None and page_num is not None:
            try:
                print(f"Pre-rendering high-res page for fast extraction (Zoom {args.capture_zoom})...")
                # Render full page at high res
                # careful with memory, but 300MB is fine for modern machines
                high_res_img = render_page(doc, page_num, zoom=args.capture_zoom)
                if high_res_img is not None:
                    # Convert to grayscale immediately to save space
                    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2GRAY)
                    # Sharpen / Binarize to match template quality
                    _, high_res_img = cv2.threshold(high_res_img, 180, 255, cv2.THRESH_BINARY)
                    zoom_ratio = args.capture_zoom / args.zoom
                    print("High-res page rendered.")
                else:
                    print("Failed to render high-res page. Using fallback.")
            except Exception as e:
                print(f"High-res pre-render failed: {e}. Using fallback.")
        
        # 4. Match Candidates
        matches = []
        import time
        start_time = time.time()
        
        for idx, cand in enumerate(candidates):
            if idx % 100 == 0:
                print(f"Matching {idx}/{len(candidates)}...", end='\r')
                
            roi = None
            
            if high_res_img is not None:
                # Fast Crop Strategy
                try:
                    x, y, w, h = cand['bbox']
                    
                    # Scale coordinates to high-res space
                    x_hi = int(x * zoom_ratio)
                    y_hi = int(y * zoom_ratio)
                    w_hi = int(w * zoom_ratio)
                    h_hi = int(h * zoom_ratio)
                    
                    # Ensure bounds
                    h_img, w_img = high_res_img.shape
                    x_hi = max(0, min(x_hi, w_img - 1))
                    y_hi = max(0, min(y_hi, h_img - 1))
                    # Add small padding?
                    pad = 2
                    x_hi = max(0, x_hi - pad)
                    y_hi = max(0, y_hi - pad)
                    w_hi = min(w_img - x_hi, w_hi + 2*pad)
                    h_hi = min(h_img - y_hi, h_hi + 2*pad)
                    
                    if w_hi > 0 and h_hi > 0:
                        roi = high_res_img[y_hi:y_hi+h_hi, x_hi:x_hi+w_hi]
                except Exception as e:
                    pass
            
            # Fallback if ROI is still None (e.g. no PDF, or error)
            if roi is None:
               roi = extract_candidate_roi(gray_img, cand)
            
            if roi is None or roi.size == 0:
                continue

            char, score = match_character(roi, tm.templates)
            
            if score > 0.5: # Threshold
                match_data = cand.copy()
                match_data['char'] = char
                match_data['score'] = score
                matches.append(match_data)
        
        elapsed = time.time() - start_time
        print(f"Matching completed in {elapsed:.2f}s.")
        print(f"Matched {len(matches)} characters.")
        
        # 5. Group Labels
        labels = group_characters(matches)
        print(f"Detected {len(labels)} labels:")
        
        # Prepare Text Export
        txt_output_path = Path(args.image_path).stem + suffix + "_extracted.txt"
        with open(txt_output_path, "w") as f:
            f.write(f"Source: {args.image_path}\n")
            if page_num is not None:
                f.write(f"Page: {page_num + 1}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Label':<15} | {'Confidence':<10} | {'BBox (x,y,w,h)'}\n")
            f.write("-" * 40 + "\n")
            
            for l in labels:
                print(f" - {l['text']} (Score: {l['score']:.2f}) at {l['bbox']}")
                # Write to file
                f.write(f"{l['text']:<15} | {l['score']:.2f}       | {l['bbox']}\n")
                
        print(f"Saved extracted text to {txt_output_path}")

        # 6. Visualize
        vis_img = orig_img.copy()
        for l in labels:
            x, y, w, h = l['bbox']
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_img, l['text'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Save output image
        out_path = Path(args.image_path).stem + suffix + "_result.png"
        cv2.imwrite(out_path, vis_img)
        print(f"Saved visualization to {out_path}")
        
        # Final Visualization Window
        # Resize if too large to fit on screen
        h_vis, w_vis = vis_img.shape[:2]
        max_h = 900
        if h_vis > max_h:
            scale = max_h / h_vis
            vis_disp = cv2.resize(vis_img, None, fx=scale, fy=scale)
        else:
            vis_disp = vis_img
            
        window_title = f"Result: {suffix if suffix else 'Image'}"
        cv2.imshow(window_title, vis_disp)
        print("Press any key to continue/exit...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_title)

if __name__ == "__main__":
    main()
