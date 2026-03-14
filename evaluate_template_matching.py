#!/usr/bin/env python3
"""
Evaluate template matching accuracy per character on selectable PDFs.

For each character found in the PDF's selectable text layer that has a
corresponding template, renders the character region at high resolution
and runs template matching. Outputs per-character accuracy and a full
confusion matrix as CSV.

Usage:
    python evaluate_template_matching.py path/to/file.pdf \
        --templates schematics_templates \
        --pages 1,3,5 \
        --output eval_results.csv
"""

import argparse
import sys
import os
import csv
import fitz
import cv2
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_manager import TemplateManager
from matcher import match_character


def extract_characters_from_page(doc, page_num):
    """Extract individual characters with bounding boxes from a PDF page."""
    page = doc.load_page(page_num)
    data = page.get_text("rawdict")

    chars = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch["c"]
                    if len(c) != 1 or c.isspace():
                        continue
                    chars.append({
                        "char": c,
                        "bbox": ch["bbox"],
                    })
    return chars


def render_char_image(doc, page_num, bbox, zoom=24.0, padding_pts=1.0):
    """Render a single character region from the PDF at high resolution."""
    x0, y0, x1, y1 = bbox
    x0 -= padding_pts
    y0 -= padding_pts
    x1 += padding_pts
    y1 += padding_pts

    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    clip = fitz.Rect(x0, y0, x1, y1)
    pix = page.get_pixmap(matrix=mat, clip=clip)

    if pix.h < 3 or pix.w < 2:
        return None

    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 4:
        img_data = img_data.reshape((pix.h, pix.w, 4))
        gray = cv2.cvtColor(img_data, cv2.COLOR_RGBA2GRAY)
    elif pix.n == 3:
        img_data = img_data.reshape((pix.h, pix.w, 3))
        gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_data.reshape((pix.h, pix.w))

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate template matching accuracy per character on a selectable PDF."
    )
    parser.add_argument("pdf_path", help="Path to a PDF with selectable text")
    parser.add_argument("--templates", required=True,
                        help="Path to template directory (e.g. schematics_templates)")
    parser.add_argument("--zoom", type=float, default=24.0,
                        help="Zoom for rendering characters (default: 24)")
    parser.add_argument("--padding", type=float, default=1.0,
                        help="Padding in PDF points around each character bbox (default: 1.0)")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated 1-based page numbers, or 'all' (default: all)")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Min score to accept a prediction (default: 0.0 = always pick best)")
    parser.add_argument("--output", type=str, default="template_matching_eval.csv",
                        help="Output CSV path (default: template_matching_eval.csv)")
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="Save mismatch character images here for inspection")
    parser.add_argument("--digits", action="store_true",
                        help="Also evaluate digit characters 0-9 (default: uppercase letters only)")
    args = parser.parse_args()

    # --- Load templates ---
    tm = TemplateManager()
    tm.load_templates_from_dir(args.templates)
    if not tm.templates:
        print("No templates loaded. Check the template directory.")
        return

    letter_templates = {k: v for k, v in tm.templates.items() if k.isalpha() and k.isupper()}
    digit_templates = {k: v for k, v in tm.templates.items() if k.isdigit()}

    if args.digits:
        eval_templates = {**letter_templates, **digit_templates}
    else:
        eval_templates = letter_templates

    available = sorted(eval_templates.keys())
    print(f"Evaluating against {len(available)} templates: {', '.join(available)}")

    # --- Open PDF ---
    doc = fitz.open(args.pdf_path)
    print(f"PDF: {args.pdf_path}  ({len(doc)} pages)")

    if args.pages is None or args.pages.lower() == "all":
        page_indices = list(range(len(doc)))
    else:
        page_indices = [int(p) - 1 for p in args.pages.split(",")]

    # --- Per-character stats ---
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    total_chars = 0
    total_correct = 0
    skipped_no_match = 0

    for page_idx in page_indices:
        print(f"\n--- Page {page_idx + 1} ---")
        chars = extract_characters_from_page(doc, page_idx)

        testable = [c for c in chars if c["char"] in eval_templates]
        print(f"  {len(chars)} characters on page, {len(testable)} testable")

        for i, ch_info in enumerate(testable):
            gt_char = ch_info["char"]
            bbox = ch_info["bbox"]

            char_img = render_char_image(doc, page_idx, bbox,
                                         zoom=args.zoom, padding_pts=args.padding)
            if char_img is None or char_img.size == 0:
                skipped_no_match += 1
                continue

            pred_char, score, _ = match_character(
                char_img, tm.templates, rotation_angles=[0]
            )

            if pred_char is None or score < args.score_threshold:
                pred_char = "?"
                skipped_no_match += 1
                stats[gt_char]["total"] += 1
                stats[gt_char]["predictions"]["?"] += 1
                total_chars += 1
                continue

            is_correct = (pred_char == gt_char)

            stats[gt_char]["total"] += 1
            stats[gt_char]["predictions"][pred_char] += 1
            if is_correct:
                stats[gt_char]["correct"] += 1

            total_chars += 1
            if is_correct:
                total_correct += 1

            if args.debug_dir and not is_correct:
                fname = f"p{page_idx+1}_{i}_gt_{gt_char}_pred_{pred_char}_{score:.2f}.png"
                cv2.imwrite(os.path.join(args.debug_dir, fname), char_img)

        if total_chars > 0:
            print(f"  Running accuracy: {total_correct}/{total_chars}"
                  f" ({100 * total_correct / total_chars:.1f}%)")

    # --- Print results table ---
    print("\n" + "=" * 72)
    print("TEMPLATE MATCHING EVALUATION RESULTS")
    print("=" * 72)
    print(f"{'Char':<6} {'Total':<8} {'Correct':<10} {'Accuracy%':<12} {'Top Confusion'}")
    print("-" * 72)

    for char in sorted(stats.keys()):
        s = stats[char]
        acc = 100.0 * s["correct"] / s["total"] if s["total"] > 0 else 0.0

        wrong = {k: v for k, v in s["predictions"].items() if k != char}
        if wrong:
            top_wrong = max(wrong, key=wrong.get)
            conf_str = f"{top_wrong} ({wrong[top_wrong]}x)"
        else:
            conf_str = "-"

        print(f"{char:<6} {s['total']:<8} {s['correct']:<10} {acc:<11.1f}% {conf_str}")

    print("-" * 72)
    overall = 100.0 * total_correct / total_chars if total_chars > 0 else 0.0
    print(f"{'TOTAL':<6} {total_chars:<8} {total_correct:<10} {overall:<11.1f}%")
    if skipped_no_match:
        print(f"  ({skipped_no_match} characters returned no valid match)")
    print("=" * 72)

    # --- Write CSV ---
    all_pred_chars = sorted(set(
        p for s in stats.values() for p in s["predictions"].keys()
    ))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Character", "Total", "Correct", "Accuracy%"] + \
                 [f"→{c}" for c in all_pred_chars]
        writer.writerow(header)

        for char in sorted(stats.keys()):
            s = stats[char]
            acc = 100.0 * s["correct"] / s["total"] if s["total"] > 0 else 0.0
            row = [char, s["total"], s["correct"], f"{acc:.1f}"]
            for pc in all_pred_chars:
                row.append(s["predictions"].get(pc, 0))
            writer.writerow(row)

        writer.writerow(["TOTAL", total_chars, total_correct, f"{overall:.1f}"])

    print(f"\nCSV saved to: {args.output}")
    doc.close()


if __name__ == "__main__":
    main()


'''
cd text_extractor

# Uppercase letters only, using schematics templates
python evaluate_template_matching.py /path/to/selectable.pdf \
    --templates schematics_templates \
    --pages 1,2,3 \
    --output eval_results.csv

# Include digits too
python evaluate_template_matching.py /path/to/selectable.pdf \
    --templates schematics_templates \
    --digits \
    --output eval_results.csv

# Save debug images of mismatches for inspection
python evaluate_template_matching.py /path/to/selectable.pdf \
    --templates schematics_templates \
    --debug-dir debug_eval \
    --output eval_results.csv
'''