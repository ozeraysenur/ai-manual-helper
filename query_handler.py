import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

load_dotenv()

COLLECTION_NAME = "manual_chunks"
CHROMA_DB_PATH = "./chroma_db"


def initialize_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client, embedding_function


def get_or_create_collection(client, embedding_function):
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"Found existing collection: {COLLECTION_NAME}")
        print(f"Collection has {collection.count()} documents")
        return collection
    except chromadb.errors.NotFoundError:
        print(f"Collection {COLLECTION_NAME} not found. Creating new one...")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"Created new collection: {COLLECTION_NAME}")
        return collection


def check_collection_contents(collection):
    count = collection.count()
    print(f"\nCollection Statistics:")
    print(f"   - Total documents: {count}")
    if count > 0:
        results = collection.get(limit=3)
        print(f"   - Sample document IDs: {results['ids'][:3]}")
        if results['metadatas']:
            print(f"   - Sample metadata: {results['metadatas'][0]}")
    return count > 0


def retrieve_relevant_chunks(query, product=None, top_k=8):  # Daha fazla chunk al
    client, embedding_function = initialize_chromadb()
    collection = get_or_create_collection(client, embedding_function)

    if not check_collection_contents(collection):
        print("Collection is empty! Please run document processing first.")
        return []

    try:
        where_clause = None
        if product:
            where_clause = {"source": {"$eq": product}}
            print(f" Filtering by product: {product}")

        print(f" Searching for: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        if not results['documents'] or not results['documents'][0]:
            print(" No relevant chunks found for your query.")
            return []

        chunks = []
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        )):
            similarity_score = 1 - distance
            chunks.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': similarity_score,
                'rank': i + 1
            })
            print(
                f" Chunk {i + 1}: Similarity={similarity_score:.3f}, Source={metadata.get('source', 'N/A')}, Page={metadata.get('page', 'N/A')}")

        # Yüksek benzerlik skoruna sahip chunk'ları filtrele
        high_quality_chunks = [chunk for chunk in chunks if chunk['similarity_score'] > 0.3]

        if not high_quality_chunks:
            print("⚠️ No high-quality matches found. Using best available chunks.")
            return chunks[:5]  # En iyi 5'ini al

        return high_quality_chunks[:6]  # En iyi 6'sını al

    except Exception as e:
        print(f" Error during retrieval: {str(e)}")
        return []


def generate_gemini_answer(query, chunks):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return " Gemini API anahtarı bulunamadı. Lütfen .env dosyasına GEMINI_API_KEY=your_key_here şeklinde ekleyin."

    try:
        genai.configure(api_key=api_key)
        # Doğru model adını kullan
        model = genai.GenerativeModel("gemini-1.5-flash")  # gemini-2.5-flash yerine

        # Context'i hazırla - daha kısa ve öz
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # En fazla 5 chunk kullan
            source = chunk['metadata'].get('source', 'Bilinmeyen')
            page = chunk['metadata'].get('page', 'N/A')
            content = chunk['content'][:800]  # Her chunk'ı kısalt

            context_parts.append(f"[Kaynak {i + 1}: {source} - Sayfa {page}]\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        # Daha kısa ve etkili prompt
        prompt = f"""Sen bir teknik dokümantasyon uzmanısın. Sana verilen manual bilgilerini kullanarak kullanıcının sorusunu yanıtla.

KULLANICI SORUSU: {query}

MANUAL BİLGİLERİ:
{context}

TALİMATLAR:
1. Soruyu manual bilgilerine göre yanıtla
2. Eğer tam cevap yoksa, bulunan ilgili bilgileri paylaş
3. Hangi kaynaktan bilgi aldığını belirt
4. Kullanıcıya ek yönlendirme yap
5. Türkçe yanıtla
6. Kısa ve öz ol (maksimum 500 kelime)

YANIT:"""

        # Gemini'den cevap al
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=800,  # Daha kısa yanıtlar
            )
        )

        if response.text:
            return response.text
        else:
            return " Gemini'den yanıt alınamadı. Lütfen tekrar deneyin."

    except Exception as e:
        error_msg = str(e)
        print(f" Gemini API hatası detayı: {error_msg}")

        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return " Geçersiz API anahtarı. Lütfen Gemini API anahtarınızı kontrol edin."
        elif "QUOTA_EXCEEDED" in error_msg or "quota" in error_msg.lower():
            return " API kotanız dolmuş. Lütfen daha sonra tekrar deneyin."
        elif "SAFETY" in error_msg or "safety" in error_msg.lower():
            return " İçerik güvenlik politikalarına takıldı. Lütfen sorunuzu yeniden formüle edin."
        elif "not found" in error_msg.lower() or "model" in error_msg.lower():
            return " Gemini model hatası. Lütfen model adını kontrol edin (gemini-1.5-flash kullanılıyor)."
        else:
            return f" Gemini API hatası: {error_msg}"


def test_gemini_connection():
    """Gemini API bağlantısını test et"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(" GEMINI_API_KEY bulunamadı!")
        return False

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # Doğru model adı
        response = model.generate_content("Test mesajı - kısa yanıt ver")
        if response.text:
            print(" Gemini API bağlantısı başarılı!")
            print(f" Test cevabı: {response.text[:100]}...")
            return True
        else:
            print(" Gemini'den yanıt alınamadı")
            return False
    except Exception as e:
        print(f" Gemini API bağlantı hatası: {str(e)}")
        return False


def generate_answer(query, chunks):
    if not chunks:
        return """ Sorunuzla ilgili bilgi bulunamadı. 

🔧 Öneriler:
• Sorunuzu farklı kelimelerle tekrar deneyin
• Daha genel terimler kullanın
• Dokümanların doğru işlendiğinden emin olun
• Belirli bir ürün/manual adı belirtin"""

    print(f" Gemini ile yanıt oluşturuluyor...")
    return generate_gemini_answer(query, chunks)


def list_available_manuals():
    client, embedding_function = initialize_chromadb()
    try:
        collection = get_or_create_collection(client, embedding_function)
        all_docs = collection.get()
        if all_docs['metadatas']:
            sources = set()
            for metadata in all_docs['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])

            if sources:
                print(" Available manuals:")
                for source in sorted(sources):
                    print(f"   📖 {source}")
                return list(sources)
            else:
                print(" No manuals found in database.")
                return []
        else:
            print(" No manuals found in database.")
            return []
    except chromadb.errors.NotFoundError:
        print(" No collection found. Please process documents first.")
        return []


def main():
    print(" Manual Assistant - Query Handler")
    print("=" * 50)

    # İlk olarak Gemini API bağlantısını test et
    print("🔍 Gemini API bağlantısı test ediliyor...")
    if not test_gemini_connection():
        print("\n Gemini API ile bağlantı kurulamadı!")
        print("\n Lütfen şunları kontrol edin:")
        print("1. .env dosyasında GEMINI_API_KEY doğru tanımlı mı?")
        print("2. API anahtarınız geçerli mi?")
        print("3. İnternet bağlantınız var mı?")
        print("4. gemini-1.5-flash modeli kullanılıyor mu?")
        return

    available_manuals = list_available_manuals()
    if not available_manuals:
        print("\n No processed manuals found!")
        print("Please run document processing first to add manuals to the database.")
        return

    print("\n" + "=" * 50)
    while True:
        query = input("\n Sorunuzu girin (çıkmak için 'q'): ").strip()
        if query.lower() in ['q', 'quit', 'exit', 'çık']:
            print(" Görüşmek üzere!")
            break

        if not query:
            print(" Lütfen geçerli bir soru girin.")
            continue

        print(f"\n🔍 İşleniyor: '{query}'")
        product = input(" (İsteğe bağlı) Belirli bir manual/ürün adı (boş bırakabilirsiniz): ").strip()

        if product and product not in available_manuals:
            print(f"⚠️  Manual '{product}' bulunamadı. Tüm manuallerde aranacak.")
            product = None

        print(f"\n İlgili bilgiler aranıyor...")
        chunks = retrieve_relevant_chunks(query, product=product)

        if chunks:
            print(f"\n {len(chunks)} adet ilgili bilgi bulundu")
            print(f"\nCevap oluşturuluyor...")
            answer = generate_answer(query, chunks)
            print("\n" + "=" * 80)
            print("📋 CEVAP:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
        else:
            print("\nİlgili bilgi bulunamadı.")
            print("\nÖneriler:")
            print("• Sorunuzu farklı kelimelerle tekrar deneyin")
            print("• Daha genel terimler kullanın")
            print("• Belirli bir ürün/konu belirtin")


if __name__ == "__main__":
    main()