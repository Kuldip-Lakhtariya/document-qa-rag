# Document Q&A — RAG-Based PDF Assistant

Upload any PDF and ask questions about it in plain language. Built as a Retrieval-Augmented Generation (RAG) pipeline — the model never sees the whole document, only the specific chunks relevant to each question, retrieved fresh every time.

**Live app:** https://document-qa-rag-9zes.onrender.com/
**Repo:** https://github.com/Kuldip-Lakhtariya/document-qa-rag

> Note: hosted on Render's free tier — the app may take 30–60 seconds to wake up on first load after inactivity.

---

## What it does

1. Upload a PDF through the browser.
2. The document is extracted, chunked, and embedded locally.
3. Ask a question — the app retrieves the most relevant chunks and asks an LLM to answer using only that context.
4. Ask as many follow-up questions as you like; each is answered fresh against the same indexed document.

---

## Architecture

```
PDF upload
   │
   ▼
extract_text.py    → raw text, tracked per page (pdfplumber)
   │
   ▼
chunker.py          → overlapping chunks, chunked per page (500 chars, 50 overlap)
   │
   ▼
embedder.py         → local embeddings (sentence-transformers, all-MiniLM-L6-v2, 384-dim)
   │
   ▼
vectordb.py          → FAISS index (flat L2) + parallel chunk lookup

────────────────────────────────────────────────

User question
   │
   ▼
embedder.py         → question embedded with the same model
   │
   ▼
vectordb.py          → similarity search, top-k chunks retrieved
   │
   ▼
generator.py         → Gemini API generates an answer from retrieved chunks only
```

Retrieval (finding relevant chunks) is pure math — cosine/L2 similarity search, no LLM involved. Generation (writing the final answer) is the only step that calls an LLM, and only ever sees the top-k retrieved chunks, never the full document.

---

## Tech stack

| Stage | Tool | Why |
|---|---|---|
| PDF extraction | `pdfplumber` | Native page-by-page extraction — preserves page numbers for citation |
| Chunking | Custom (fixed-size + overlap) | Simple, predictable, per-page to protect page-number accuracy |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local, CPU-friendly — avoids per-chunk API cost at indexing time |
| Vector store | FAISS (`IndexFlatL2`) | Built from scratch (not Chroma) deliberately, to learn the retrieval mechanics directly |
| Generation | Google Gemini API (`gemini-3.5-flash`) | Free tier, no ongoing cost |
| Backend | Flask + Gunicorn | Consistent with prior projects' deployment pattern |
| Deployment | Docker + Render (free tier) | CPU-only PyTorch build used to keep image size down |

---

## Key design decisions

- **Chunking happens per-page, not across the whole document** — this guarantees every chunk stays attributable to exactly one page number, which matters for showing users where an answer came from.
- **Overlap between chunks** (50 characters) ensures an idea cut at a chunk boundary still appears whole in at least one chunk, rather than being permanently fractured.
- **Local embeddings, API-based generation** — embeddings run far more frequently (once per chunk, every upload) than generation (once per question), so keeping embeddings free and local was the right cost/latency tradeoff.
- **FAISS over Chroma** — chosen intentionally to build and understand the index + metadata-lookup mechanics manually, rather than relying on an abstraction, since this is a learning-focused project.
- **CPU-only PyTorch in the Docker build** — `sentence-transformers` installs GPU-enabled PyTorch by default; since this app never uses a GPU, the CPU-only build is installed explicitly first to avoid a bloated image.

---

## Known limitations (by design, for now)

This is v1 of the project — deployed working end-to-end first, with the following gaps intentionally deferred rather than solved upfront:

- **Single document at a time.** The vector index is rebuilt fresh on every upload; a new upload replaces the previous document rather than supporting multiple documents or multiple users simultaneously.
- **Broad/summary questions are weaker than specific questions.** Basic similarity-search RAG retrieves only the top-k most relevant chunks. Narrow factual questions ("what is the marks distribution for X?") work well. Broad questions ("summarize this document," "how many chapters are there?") are answered from a partial slice of the document, not the whole thing, since the answer isn't localized to a few chunks.
- **No conversation memory.** Each question is answered independently — the chat interface shows a visual history, but past questions/answers are not resent to the model, so it cannot use earlier turns as context for the current one.
- **Occasional `503` errors from Gemini.** During high demand, Gemini's API can return a temporary `503 UNAVAILABLE` — the backend now returns this as a clean JSON error rather than crashing, but the frontend does not yet auto-retry; the user has to manually ask again.
- **No file-type/size validation on upload** yet — a production version would validate the uploaded file is genuinely a PDF and within a reasonable size limit before processing.

Planned for the next iteration: multi-document/session support, more robust error handling with retries, and detection of broad vs. narrow questions to adjust retrieval strategy accordingly.

---

## Running locally

```bash
git clone https://github.com/Kuldip-Lakhtariya/document-qa-rag.git
cd document-qa-rag
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

Run:
```bash
python app.py
```
Visit `http://127.0.0.1:5000`.

---

## Author

Kuldip Lakhtariya
B.Tech ECE, LD College of Engineering, Ahmedabad

- GitHub: [Kuldip-Lakhtariya](https://github.com/Kuldip-Lakhtariya)
- LinkedIn: [kuldip-lakhtariya](https://www.linkedin.com/in/kuldip-lakhtariya-957106371/)
- HuggingFace: [kuldip2611](https://huggingface.co/kuldip2611)
- Email: kuldip2611lakhtariya@gmail.com
