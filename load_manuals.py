import os
from PyPDF2 import PdfReader

SAMPLE_MANUALS_DIR = 'sample_manuals'
CHUNK_SIZE = 200  # words


def extract_text_by_page(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({'page_num': i + 1, 'text': text})
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_manuals():
    manuals = [f for f in os.listdir(SAMPLE_MANUALS_DIR) if f.lower().endswith('.pdf')]
    all_chunks = []
    for manual in manuals:
        pdf_path = os.path.join(SAMPLE_MANUALS_DIR, manual)
        pages = extract_text_by_page(pdf_path)
        for page in pages:
            page_num = page['page_num']
            text = page['text']
            # Optionally, split by headings here (future improvement)
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'manual': manual,
                    'page_num': page_num,
                    'chunk_id': f"{manual}_p{page_num}_c{idx}",
                    'text': chunk
                })
    return all_chunks


if __name__ == "__main__":
    chunks = process_manuals()
    print(f"Extracted {len(chunks)} chunks from manuals.")
    # Optionally, print a sample chunk
    if chunks:
        print("Sample chunk:", chunks[0]) 