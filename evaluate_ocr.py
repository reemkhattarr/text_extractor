import argparse
import sys
from pathlib import Path
from collections import Counter
import re

# Import local modules
# Ensure we can import from the same directory
sys.path.append(str(Path(__file__).parent))
from pdf_loader import load_pdf, extract_words_from_page
try:
    from grouper import _classify_label
except ImportError:
    # Fallback if cannot import private function, though it should work
    def _classify_label(text):
        if re.match(r'^[A-Z]{1,2}[0-9]{1,3}$', text):
            return "COMPONENT"
        return "OTHER"

def parse_ocr_file(filepath):
    """
    Parses the OCR output text file.
    Assumes format:
    Label           | Type  | ...
    R8              | CP    | ...
    """
    labels = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_parsing = False
        for line in lines:
            line = line.strip()
            # Detect header separator or header
            if "Label" in line and "|" in line:
                start_parsing = True
                continue
            if set(line) <= set("- "): # Separator line
                continue
            
            if start_parsing and "|" in line:
                parts = line.split("|")
                label = parts[0].strip()
                if label:
                    labels.append(label)
    except Exception as e:
        print(f"Error reading OCR file: {e}")
        
    return labels

def calculate_metrics(ocr_labels, gt_labels):
    """
    Calculates Precision, Recall, F1, and Accuracy.
    Args:
        ocr_labels: list of strings
        gt_labels: list of strings
    """
    # Normalize? 
    # The OCR labels are upper case usually. 
    # PDF text might be mixed? Assuming Case Sensitive for now as component names like 'u1' vs 'U1' matter usually, but standard is Upper.
    # Let's force upper for comparison to be generous.
    ocr_norm = [x.upper() for x in ocr_labels]
    gt_norm = [x.upper() for x in gt_labels]
    
    ocr_counter = Counter(ocr_norm)
    gt_counter = Counter(gt_norm)
    
    # Intersection (True Positives)
    # intersection of multisets: min(count_a, count_b)
    tp = 0
    common_elements = list((ocr_counter & gt_counter).elements())
    tp = len(common_elements)
    
    # False Positives: In OCR but not in GT
    fp_elements = list((ocr_counter - gt_counter).elements())
    fp = len(fp_elements)
    
    # False Negatives: In GT but not in OCR
    fn_elements = list((gt_counter - ocr_counter).elements())
    fn = len(fn_elements)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy (Jaccard Index) = Intersection / Union
    union_count = tp + fp + fn
    accuracy = tp / union_count if union_count > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy,
        'FP_List': fp_elements,
        'FN_List': fn_elements
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR results against PDF Ground Truth text.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--page", required=True, type=int, help="Page number (1-based)")
    parser.add_argument("--ocr-results", required=True, help="Path to OCR result text file")
    parser.add_argument("--filter-components", action="store_true", help="If set, only evaluates labels that look like components (e.g. R1, C10)")
    parser.add_argument("--output-file", help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # 1. Load Ground Truth
    doc = load_pdf(args.pdf)
    if not doc:
        print("Could not load PDF.")
        return
        
    print(f"Extracting text from Page {args.page}...")
    gt_words = extract_words_from_page(doc, args.page - 1)
    print(f"Found {len(gt_words)} words in PDF.")
    
    # 2. Load OCR Results
    ocr_labels = parse_ocr_file(args.ocr_results)
    print(f"Loaded {len(ocr_labels)} labels from OCR result file.")
    
    # Optional Filtering
    if args.filter_components:
        # Re-use classification logic if possible or simple regex
        print("Filtering for component-like labels only...")
        gt_words = [w for w in gt_words if _classify_label(w) == "COMPONENT"]
        ocr_labels = [l for l in ocr_labels if _classify_label(l) == "COMPONENT"]
        print(f"After filtering: GT={len(gt_words)}, OCR={len(ocr_labels)}")

    # 3. Calculate Metrics
    metrics = calculate_metrics(ocr_labels, gt_words)
    
    output_lines = []
    output_lines.append("="*40)
    output_lines.append("EVALUATION RESULTS")
    output_lines.append("="*40)
    output_lines.append(f"Precision: {metrics['Precision']:.4f}")
    output_lines.append(f"Recall:    {metrics['Recall']:.4f}")
    output_lines.append(f"F1 Score:  {metrics['F1']:.4f}")
    output_lines.append(f"Accuracy:  {metrics['Accuracy']:.4f} (Jaccard)")
    output_lines.append("-" * 40)
    output_lines.append(f"True Positives:  {metrics['TP']}")
    output_lines.append(f"False Positives: {metrics['FP']} (Extra detections)")
    output_lines.append(f"False Negatives: {metrics['FN']} (Missed)")
    output_lines.append("="*40)
    
    if metrics['FP'] > 0:
        output_lines.append("\nFalse Positives (Top 20):")
        output_lines.append(", ".join(metrics['FP_List'][:20]) + ("..." if metrics['FP'] > 20 else ""))
        
    if metrics['FN'] > 0:
        output_lines.append("\nFalse Negatives (Missed) (Top 20):")
        output_lines.append(", ".join(metrics['FN_List'][:20]) + ("..." if metrics['FN'] > 20 else ""))

    result_text = "\n".join(output_lines)
    print(result_text)
    
    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(result_text)
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
