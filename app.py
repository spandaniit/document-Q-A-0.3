import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import openai
import numpy as np

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# -----------------------
# FLASK APP
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

DOC_STORE = {}  # In-memory: doc_id -> {"filename":..., "chunks":[...], "embeddings": np.array(...)}

# -----------------------
# TEXT EXTRACTION
# -----------------------
def extract_text_from_pdf(file_bytes):
    from io import BytesIO
    reader = PdfReader(BytesIO(file_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file_bytes):
    from io import BytesIO
    doc = DocxDocument(BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text)

def extract_text_from_html(file_bytes):
    html = file_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")

def extract_text(filename, data):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf": return extract_text_from_pdf(data)
    if ext == ".docx": return extract_text_from_docx(data)
    if ext in {".html", ".htm"}: return extract_text_from_html(data)
    if ext in {".txt", ".md"}: return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type")

# -----------------------
# TEXT CHUNKING & EMBEDDING
# -----------------------
def chunk_text(text, chunk_size=450, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        start = max(i - overlap, 0)
        end = min(i + chunk_size, n)
        chunks.append(" ".join(words[start:end]))
        i += chunk_size
    return chunks

def embed_texts(texts):
    resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=texts)
    return [d["embedding"] for d in resp["data"]]

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return a_norm @ b_norm

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def index():
    docs = [{"id": k, "filename": v["filename"], "n_chunks": len(v["chunks"])} for k, v in DOC_STORE.items()]
    return render_template("index.html", docs=docs)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type"}), 400
    data = f.read()
    if len(data) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large"}), 413

    try:
        text = extract_text(filename, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    chunks = chunk_text(text)
    if not chunks:
        return jsonify({"error": "Empty document"}), 400

    try:
        embeddings = np.array(embed_texts(chunks), dtype=np.float32)
    except Exception as e:
        return jsonify({"error": "Embedding failed: " + str(e)}), 500

    doc_id = str(uuid.uuid4())
    DOC_STORE[doc_id] = {"filename": filename, "chunks": chunks, "embeddings": embeddings}
    return jsonify({"status": "ok", "doc_id": doc_id, "n_chunks": len(chunks)})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    doc_id = data.get("doc_id")
    k = int(data.get("k", 4))
    if not question:
        return jsonify({"error": "Empty question"}), 400

    context_chunks = []
    if doc_id and doc_id in DOC_STORE:
        doc = DOC_STORE[doc_id]
        q_vec = np.array(embed_texts([question])[0], dtype=np.float32)
        sims = cosine_similarity_matrix(doc["embeddings"], q_vec)
        top_idx = np.argsort(-sims)[:k]
        for idx in top_idx:
            context_chunks.append(f"[Source:{doc['filename']}] {doc['chunks'][idx]}")

    if context_chunks:
        prompt = f"Answer using ONLY the context below. If answer not found, say you don't know.\n\nCONTEXT:\n\n" + "\n\n---\n\n".join(context_chunks) + f"\n\nQuestion: {question}"
    else:
        prompt = f"Answer concisely:\n\nQuestion: {question}"

    try:
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=600
        )
        answer = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": "LLM failed: " + str(e)}), 500

    return jsonify({"answer": answer})

# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
