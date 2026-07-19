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


import pdfplumber  # already a dependency via extract_text.py, now used directly for a validity check

ALLOWED_EXTENSION = ".pdf"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB — generous for typical PDFs, blocks abuse

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files["pdf"]

    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not pdf_file.filename.lower().endswith(ALLOWED_EXTENSION):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Get the real byte count from the stream itself — content_length can be
    # missing or spoofed depending on the client, so don't trust it alone.
    pdf_file.stream.seek(0, os.SEEK_END)
    file_size = pdf_file.stream.tell()
    pdf_file.stream.seek(0)  # rewind — .save() below needs to read from byte 0

    if file_size == 0:
        return jsonify({"error": "Uploaded file is empty"}), 400

    if file_size > MAX_FILE_SIZE_BYTES:
        return jsonify({"error": f"File exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit"}), 413

    save_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(save_path)

    try:
        with pdfplumber.open(save_path) as pdf_check:
            if len(pdf_check.pages) == 0:
                os.remove(save_path)
                return jsonify({"error": "PDF has no pages"}), 400
    except Exception:
        os.remove(save_path)
        return jsonify({"error": "File is corrupt or not a valid PDF"}), 400

    try:
        extracted_pages = extract_text_from_pdf(save_path)
        chunks = chunk_text(extracted_pages)
        embedded_chunks = embed_chunks(chunks)

        global vector_db
        vector_db = VectorDB(embedding_dim=384)
        vector_db.add_chunks(embedded_chunks)
    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "PDF processed", "chunks_indexed": len(embedded_chunks)})

from pipeline.question_classifier import is_broad_question

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    try:
        if is_broad_question(question):
            # Bypass retrieval — feed every indexed chunk so the model
            # actually sees the whole document, not a fraction of it.
            context_chunks = vector_db.get_all_chunks()
        else:
            query_embedding = embed_query(question)
            context_chunks = vector_db.search(query_embedding, top_k=3)

        answer = generate_answer(context_chunks, question)
    except Exception as e:
        # Covers Gemini being down, rate-limited, or any other API failure —
        # returns a clean JSON error instead of crashing into Flask's HTML page.
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 503

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=False)  
