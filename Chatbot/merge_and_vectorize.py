import json
import pickle
import uuid
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


TEXT_DATA_PATH = "structured_qa_dataset.json"  # Q&A dataset
OCR_DATA_PATH = "data.json"  # Chunks dataset
MERGED_JSON_PATH = "merged_dataset.json"
INDEX_PATH = "merged_index.faiss"
META_PATH = "merged_metadata.pkl"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


print("📂 Loading datasets...")
with open(OCR_DATA_PATH, "r", encoding="utf-8") as f:
    ocr_data = json.load(f)

with open(TEXT_DATA_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

merged_entries = []


print("🔄 Normalizing OCR/image dataset...")
for doc in ocr_data:
    for chunk in doc.get("chunks", []):
        merged_entries.append({
            "id": str(doc.get("id", uuid.uuid4())),
            "source": doc.get("source", "ocr_data"),
            "type": doc.get("type", "text"),
            "language": doc.get("language", "unknown"),
            "chunk_id": str(chunk.get("chunk_id", uuid.uuid4())),
            "text": chunk.get("text", "").strip(),
            "metadata": {k: v for k, v in doc.items() if k not in ["chunks"]}
        })

print("🔄 Normalizing Q&A dataset...")
for entry in qa_data:
    merged_entries.append({
        "id": str(uuid.uuid4()),
        "source": "qa_dataset",
        "type": entry.get("type", "qa"),
        "language": "unknown",
        "chunk_id": str(uuid.uuid4()),
        "text": entry.get("content", "").strip(),
        "metadata": {}
    })


print(f"💾 Saving merged dataset to {MERGED_JSON_PATH}...")
with open(MERGED_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(merged_entries, f, ensure_ascii=False, indent=2)

print(f"✅ Merged dataset contains {len(merged_entries)} entries.")


print(f"🔄 Loading embedding model: {EMBED_MODEL}...")
model = SentenceTransformer(EMBED_MODEL)


print("⚡ Generating embeddings...")
texts = [entry["text"] for entry in merged_entries]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
embeddings = embeddings.astype("float32")


print("📦 Creating FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"💾 Saving FAISS index to {INDEX_PATH}...")
faiss.write_index(index, INDEX_PATH)

print(f"💾 Saving metadata to {META_PATH}...")
with open(META_PATH, "wb") as f:
    pickle.dump(merged_entries, f)

print("✅ All done! You can now use merged_index.faiss and merged_metadata.pkl in app.py/query_rag.py")
