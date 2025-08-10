import faiss
import pickle
from sentence_transformers import SentenceTransformer
from ollama import chat

INDEX_PATH = "merged_index.faiss"
META_PATH = "merged_metadata.pkl"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3"
TOP_K = 5


print("🔄 Loading FAISS index & metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metas = pickle.load(f)

print("🔄 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)


def search(query, k=TOP_K):
    q_vec = embedder.encode([query]).astype("float32")
    scores, idxs = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < len(metas):
            results.append({
                "score": float(score),
                "text": metas[idx]["text"], 
                "source": metas[idx].get("source", ""),
                "type": metas[idx].get("type", ""),
                "language": metas[idx].get("language", ""),
            })
    return results


def rag_answer(question, model_name=LLM_MODEL):
    hits = search(question)

    
    print("\n📌 Retrieved context from FAISS:")
    for i, h in enumerate(hits, start=1):
        print(f"{i}. (score={h['score']:.4f}) [{h['source']}/{h['type']}] {h['text']}")

    
    context = "\n".join(f"- {h['text']}" for h in hits)

    
    system_prompt = (
        "Tu es un assistant qui répond uniquement en te basant sur le contexte suivant :\n"
        f"{context}\n\n"
        "Si l'information n'est pas disponible, dis-le clairement."
    )
    print("\n🛠️ System prompt sent to LLM:")
    print(system_prompt)
    print("=" * 60)

    response = chat(model=model_name, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])

    return response["message"]["content"], hits


if __name__ == "__main__":
    while True:
        question = input("\n❓ Question ('exit' pour quitter) : ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Fin de la session.")
            break

        answer, hits = rag_answer(question)

        print("\n🤖 Réponse :")
        print(answer)
