import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import os
from pathlib import Path
import hashlib

def process_pdf_to_chunks(pdf_path, chunk_size=500, overlap=50):
    print(f"📄 Processing PDF: {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            chunks = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    sentences = text.replace('\n', ' ').split('. ')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                chunks.append({
                                    'content': current_chunk.strip(),
                                    'page': page_num + 1,
                                    'source': os.path.basename(pdf_path)
                                })
                            current_chunk = sentence + ". "
                    if current_chunk.strip():
                        chunks.append({
                            'content': current_chunk.strip(),
                            'page': page_num + 1,
                            'source': os.path.basename(pdf_path)
                        })
                except Exception as e:
                    print(f"⚠️  Error processing page {page_num + 1}: {e}")
                    continue
            print(f"Extracted {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
            return chunks
    except Exception as e:
        print(f" Error processing PDF {pdf_path}: {e}")
        return []

def add_chunks_to_chromadb(chunks, collection_name="manual_chunks"):
    if not chunks:
        print(" No chunks to add")
        return False
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f" Using existing collection: {collection_name}")
        except chromadb.errors.NotFoundError:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"Created new collection: {collection_name}")
        documents = []
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{chunk['source']}_{chunk['page']}_{i}_{chunk['content'][:50]}".encode()
            ).hexdigest()
            documents.append(chunk['content'])
            metadatas.append({
                'source': chunk['source'],
                'page': chunk['page'],
                'chunk_index': i
            })
            ids.append(chunk_id)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(chunks)} chunks to collection")
        print(f"Collection now has {collection.count()} total documents")
        return True
    except Exception as e:
        print(f"Error adding chunks to ChromaDB: {e}")
        return False

def process_manual_directory(directory_path="data"):
    data_dir = Path(directory_path)
    if not data_dir.exists():
        print(f"Directory {directory_path} does not exist")
        return
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return
    print(f"Found {len(pdf_files)} PDF files to process")
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        chunks = process_pdf_to_chunks(str(pdf_path))
        if chunks:
            all_chunks.extend(chunks)
        else:
            print(f"No chunks extracted from {pdf_path.name}")
    if all_chunks:
        print(f"\nTotal chunks extracted: {len(all_chunks)}")
        print("Adding chunks to ChromaDB...")
        success = add_chunks_to_chromadb(all_chunks)
        if success:
            print("\n Document processing completed successfully!")
            print(" You can now run query_handler.py to ask questions")
        else:
            print("\n Failed to add chunks to database")
    else:
        print("\n No chunks were extracted from any PDF files")

if __name__ == "__main__":
    print("Document Processor for Manual Assistant")
    print("=" * 50)
    process_manual_directory("data") 