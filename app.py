"""
Simple Document Q&A (Flask) using OpenAI embeddings + Chat API.
No React/Node/Vite/Tailwind required — plain HTML templates.

How it works (in-process):
- Upload PDF / .txt / .md
- Extract text -> chunk into ~400-word pieces
- Call OpenAI embeddings for chunks, keep them in memory (numpy array)
- On question: embed query -> compute cosine similarity -> pick top-k chunks
- Build a context and call ChatCompletion -> return answer + shown sources

Note: this stores everything in RAM for the running process. For production-scale you should persist vectors.
"""

import os
import math
import uuid
from typing import List, Dict
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import openai
import numpy as np
from PyPDF2 import PdfReader

load_dotenv()  # loads .env if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change if you don't have access

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in environment or .env file")

openai.api_key = OPENAI_API_KEY

APP = Flask(__name__)
APP.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads
ALLOWED_EXT = {"pdf", "txt", "md"}

# In-memory store (simple)
DOC_STORE: Dict[str, Dict] = {
    # Example structure:
    # "doc_id": {
    #    "filename": "...",
    #    "chunks": [ "text1", "text2", ... ],
    #    "metadatas": [ {"chunk_index":0, "start_word":0, ...}, ... ],
    #    "embeddings": np.array([...])  # shape (n_chunks, dim)
    # }
}

def allowed_filename(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    # Write to temp file then read via PyPDF2
    try:
        # PyPDF2 can read from BytesIO, but to be safe use reader on bytes
        from io import BytesIO
        reader = PdfReader(BytesIO(file_bytes))
        text_parts = []
        for p in reader.pages:
            text = p.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        print("PDF extraction error:", e)
        return ""

def chunk_text_by_words(text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Returns list of {"content": str, "start_word": int, "end_word": int}
    chunk_size = approx words per chunk.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        start = max(i - overlap, 0)
        end = min(i + chunk_size, n)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({"content": chunk_text, "start_word": start, "end_word": end})
        i += chunk_size
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Use OpenAI embeddings API (batch). Returns list of vectors.
    """
    # OpenAI Python library supports multiple inputs
    resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d["embedding"] for d in resp["data"]]

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (m, d), b: (n, d) -> returns (m, n)
    # compute normalized dot product
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def get_top_k_chunks(doc: Dict, query: str, k: int = 4):
    """
    Returns list of (chunk_index, score, chunk_text, metadata)
    """
    q_emb = embed_texts([query])
    q_vec = np.array(q_emb, dtype=np.float32)
    chunk_embs = doc["embeddings"]  # numpy array (n_chunks, dim)
    sims = cosine_sim_matrix(q_vec, chunk_embs)[0]  # shape (n_chunks,)
    order = np.argsort(-sims)  # descending
    top = []
    for idx in order[:k]:
        top.append((int(idx), float(sims[idx]), doc["chunks"][idx], doc["metadatas"][idx]))
    return top

def build_prompt(context_chunks: List[str], question: str) -> str:
    ctx = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are an assistant that answers questions strictly based on the provided CONTEXT. "
        "Use the context to answer. If the answer is not present in the context, say 'I don't know' or be honest.\n\n"
        f"CONTEXT:\n{ctx}\n\nQuestion: {question}\n\nAnswer concisely and mention the source chunk numbers in square brackets like [chunk 2]."
    )
    return prompt

def call_chat_completion(system_prompt: str, user_prompt: str) -> str:
    # Use chat completions (gpt-3.5/gpt-4 family) - fallback to simple completion if needed
    try:
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Chat completion error:", e)
        # As fallback, use standard completions (less recommended)
        try:
            resp2 = openai.Completion.create(
                model="text-davinci-003",
                prompt=user_prompt,
                max_tokens=512,
                temperature=0.0,
            )
            return resp2["choices"][0]["text"].strip()
        except Exception as e2:
            print("Fallback completion error:", e2)
            return "Error: LLM call failed."

@APP.route("/", methods=["GET"])
def index():
    # Show the upload / ask forms, list uploaded docs
    docs = []
    for doc_id, info in DOC_STORE.items():
        docs.append({"id": doc_id, "filename": info["filename"], "n_chunks": len(info["chunks"])})
    return render_template("index.html", docs=docs, answer=None, sources=None)

@APP.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))
    f = request.files["file"]
    filename = secure_filename(f.filename)
    if not filename:
        return redirect(url_for("index"))
    if not allowed_filename(filename):
        return f"File type not allowed. Allowed: {', '.join(ALLOWED_EXT)}", 400

    data = f.read()
    # Extract text
    text = ""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text = extract_text_from_pdf_bytes(data)
    else:
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    if not text.strip():
        return "Could not extract text from file or file is empty.", 400

    # Chunking
    chunks_meta = chunk_text_by_words(text, chunk_size=400, overlap=80)
    chunks_texts = [c["content"] for c in chunks_meta]
    if len(chunks_texts) == 0:
        return "No chunks generated from document.", 400

    # Embeddings (batch)
    try:
        embedding_list = embed_texts(chunks_texts)
        emb_array = np.array(embedding_list, dtype=np.float32)
    except Exception as e:
        print("Embedding error:", e)
        return "Error creating embeddings: " + str(e), 500

    # Store in memory with uuid
    doc_id = str(uuid.uuid4())
    DOC_STORE[doc_id] = {
        "filename": filename,
        "chunks": chunks_texts,
        "metadatas": chunks_meta,
        "embeddings": emb_array
    }

    return redirect(url_for("index"))

@APP.route("/ask", methods=["POST"])
def ask():
    doc_id = request.form.get("doc_id")
    question = request.form.get("question", "").strip()
    k = int(request.form.get("k", 4))
    if not doc_id or doc_id not in DOC_STORE:
        return "Document not found. Upload first.", 400
    if not question:
        return "Empty question.", 400

    doc = DOC_STORE[doc_id]
    top = get_top_k_chunks(doc, question, k=k)

    # Build context pieces with chunk indices for citation
    context_pieces = []
    source_map = []
    for idx, score, chunk_text, meta in top:
        label = f"[chunk {idx}]"
        context_pieces.append(f"{label}\n{chunk_text[:2000]}")  # limit chunk length in prompt
        source_map.append({"index": idx, "score": score, "text": chunk_text})

    prompt = build_prompt(context_pieces, question)
    system_prompt = "You are a helpful assistant. Answer using only the provided context and cite chunk numbers."

    answer = call_chat_completion(system_prompt, prompt)

    # Prepare display sources (truncate)
    sources = [{"index": s["index"], "score": s["score"], "text": s["text"][:800]} for s in source_map]

    # Render same index page with answer
    docs = []
    for d_id, info in DOC_STORE.items():
        docs.append({"id": d_id, "filename": info["filename"], "n_chunks": len(info["chunks"])})

    return render_template("index.html", docs=docs, answer=answer, sources=sources, selected_doc=doc_id, question=question)

if __name__ == "__main__":
    # For local dev
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)