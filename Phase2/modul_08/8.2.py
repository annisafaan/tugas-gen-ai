# Pastikan sudah menjalankan `pip install voyageai` di terminal jika belum
import voyageai, os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Mengambil API key langsung dari .env yang sudah kamu setting[cite: 38]
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# Memanggil API embeddings menggunakan model yang direkomendasikan[cite: 38]
result = vo.embed(
    ["What is RAG?", "Explain vector databases."],
    model="voyage-3",          # current recommended model[cite: 38]
    input_type="document"      # "document" for corpus, "query" for search queries[cite: 38]
)

embeddings = np.array(result.embeddings, dtype=np.float32)
print(f"Shape: {embeddings.shape}")       # Seharusnya (2, 1024)[cite: 38]
print(f"Token usage: {result.total_tokens}")