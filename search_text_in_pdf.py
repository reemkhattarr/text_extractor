import fitz

def main():
    pdf_path = "pico-datasheet.pdf"
    target = "R8"
    
    print(f"Searching for '{target}' in {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        print(f"Total Pages: {len(doc)}")
        
        for i in range(len(doc)):
            page = doc.load_page(i)
            # Search for R8
            hits = page.search_for(target)
            if hits:
                print(f"Found '{target}' on Context Page {i+1} (Index {i}) - {len(hits)} times.")
                for rect in hits:
                     print(f"  - Rect: {rect}")
            
            # Also check text
            text = page.get_text()
            if target in text:
                 print(f"Found '{target}' in text of Context Page {i+1}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
