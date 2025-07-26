import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import load_manuals

# Load environment variables from .env
load_dotenv()

# ChromaDB setup
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "manual_chunks"

# OpenAI setup
OPENAI_EMBED_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_embedding(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=OPENAI_EMBED_MODEL
    )
    return response.data[0].embedding


def main():
    # Load chunks
    chunks = load_manuals.process_manuals()
    print(f"Loaded {len(chunks)} chunks. Generating embeddings...")

    # Setup ChromaDB
    client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        collection = client.get_collection(COLLECTION_NAME)
    else:
        collection = client.create_collection(COLLECTION_NAME)

    # Prepare data for ChromaDB
    ids = []
    texts = []
    metadatas = []
    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        texts.append(chunk['text'])
        metadatas.append({
            'manual': chunk['manual'],
            'page_num': chunk['page_num']
        })

    # Generate and store embeddings
    for i, (cid, text, meta) in enumerate(zip(ids, texts, metadatas)):
        emb = get_openai_embedding(text)
        collection.add(
            ids=[cid],
            embeddings=[emb],
            documents=[text],
            metadatas=[meta]
        )
        if (i+1) % 10 == 0 or (i+1) == len(ids):
            print(f"Embedded {i+1}/{len(ids)} chunks...")
    print("All chunks embedded and stored in ChromaDB.")
    print("Available collections after embedding:", [c.name for c in client.list_collections()])

if __name__ == "__main__":
    main() 