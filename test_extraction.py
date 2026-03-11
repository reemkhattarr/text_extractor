import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from pdf_loader import load_pdf, extract_words_from_page

def main():
    pdf_path = "pico-datasheet.pdf"
    page_num = 28
    
    print(f"Loading {pdf_path}...")
    doc = load_pdf(pdf_path)
    if not doc:
        print("Failed to load PDF")
        return

    print(f"Extracting words from Page {page_num}...")
    words = extract_words_from_page(doc, page_num - 1)
    
    with open("extracted_vectors.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    print(f"Saved {len(words)} words to extracted_vectors.txt")

if __name__ == "__main__":
    main()
