from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os

load_dotenv()  # loads GEMINI_API_KEY from .env into os.environ, locally

from pipeline.extract_text import extract_text_from_pdf
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks, embed_query
from pipeline.vectordb import VectorDB
from pipeline.generator import generate_answer


app = Flask(__name__)

# supports ONE uploaded document at a time across ALL users — a real
# limitation to flag now, fix it later.
vector_db = VectorDB(embedding_dim=384)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files["pdf"]
    save_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(save_path)

    try:
        extracted_pages = extract_text_from_pdf(save_path)
        chunks = chunk_text(extracted_pages)
        embedded_chunks = embed_chunks(chunks)

        # Rebuild the index fresh for each new upload — since we only
        # support one active document right now (flagged above).
        global vector_db
        vector_db = VectorDB(embedding_dim=384)
        vector_db.add_chunks(embedded_chunks)

    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "PDF processed", "chunks_indexed": len(embedded_chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    try:
        query_embedding = embed_query(question)
        top_chunks = vector_db.search(query_embedding, top_k=3)
        answer = generate_answer(top_chunks, question)
    except Exception as e:
        # Covers Gemini being down, rate-limited, or any other API failure —
        # returns a clean JSON error instead of crashing into Flask's HTML page.
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 503

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)  # debug=True for LOCAL testing only — must be False in production