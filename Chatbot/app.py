from flask import Flask, request, Response, send_from_directory
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from ollama import chat
import re


INDEX_PATH = "merged_index.faiss"
META_PATH = "merged_metadata.pkl"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3"
TOP_K = 5
RELEVANCE_THRESHOLD = 8.0  

app = Flask(__name__)


print("🔄 Loading FAISS index & metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metas = pickle.load(f)

print("🔄 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)


def search_context(query, k=TOP_K):
    q_vec = embedder.encode([query]).astype("float32")
    scores, idxs = index.search(q_vec, k)

    print("\n📌 FAISS search results for query:", query)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < len(metas):
            text = metas[idx]["text"].strip()
            if text:
                print(f"  Score={score:.4f} | Text={text[:80]}...")
                results.append({"score": float(score), "text": text})
    return results


def clean_context(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return {"error": "No question provided"}, 400

    
    context_chunks = search_context(question)
    cleaned_chunks = [clean_context(c["text"]) for c in context_chunks]
    context_text = "\n".join(f"- {c}" for c in cleaned_chunks) if cleaned_chunks else ""

    
    use_context = False
    if context_chunks and context_chunks[0]["score"] <= RELEVANCE_THRESHOLD:
        use_context = True

   
    if use_context:
        system_prompt = (
            "Tu es un assistant qui répond en utilisant prioritairement les informations du contexte suivant.\n"
            "Si nécessaire, complète avec tes connaissances générales.\n"
            "Si un calcul est nécessaire, effectue-le.\n"
            "Réponds de manière concise et précise.\n\n"
            f"Contexte:\n{context_text}"
        )
    else:
        system_prompt = (
            "Tu es un assistant intelligent capable de répondre à des questions en utilisant tes connaissances générales "
            "et en effectuant des calculs si nécessaire. "
            "Si la question concerne les documents internes, utilise le contexte fourni (s'il est pertinent).\n\n"
            f"Contexte:\n{context_text if context_text else '(Pas de contexte pertinent)'}"
        )

   
    def stream():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        response = chat(model=LLM_MODEL, messages=messages, stream=True)

        collected = ""
        for chunk in response:
            token = chunk["message"]["content"]
            collected += token
            yield token

    return Response(stream(), content_type="text/plain")


@app.route("/")
def serve_ui():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


if __name__ == "__main__":
    app.run(debug=True)
