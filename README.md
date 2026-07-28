# Document Q&A — RAG-Based PDF Assistant

Upload any PDF and ask questions about it in plain language. Built as a Retrieval-Augmented Generation (RAG) pipeline — retrieval finds the relevant chunks, generation answers using only that context (except for broad/summary questions — see Part 2).

**Repo:** https://github.com/Kuldip-Lakhtariya/document-qa-rag

**Live app(P1):** https://document-qa-rag-9zes.onrender.com/

**Live app(P2):** https://document-qa-rag-1-08ry.onrender.com/

> ⚠️ **Known constraint, not a bug:** hosted on Render's free tier (512MB RAM cap). Repeated large uploads within the same long-running session can push memory toward that limit over time — see "Memory behavior" under Known Limitations for the actual mechanism (it's more subtle than simple file size).
> Also: free-tier services spin down after 15 min idle — first load after a gap can take 30–60s.

---

## Repo structure

```
document-qa-rag/
├── pipeline/
│   ├── __init__.py
│   ├── extract_text.py         # PDF → page-tracked raw text (pdfplumber)
│   ├── chunker.py              # page-tracked text → overlapping chunks
│   ├── embedder.py             # chunks/questions → embeddings (sentence-transformers)
│   ├── vectordb.py             # FAISS index + chunk lookup, per-session
│   ├── question_classifier.py # broad vs. narrow question detection
│   └── generator.py            # Gemini API call, retry logic, Groq fallback
├── templates/
│   └── index.html              # frontend — upload, chat log, feedback
├── uploads/                    # saved PDFs — gitignored
├── app.py                      # routes: /, /upload, /ask, /feedback
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Part 1 — Initial build

First working version, built end-to-end: upload → index → ask → answer.

**What it does:**
- Upload a PDF → extracted, chunked, embedded locally
- Ask a question → top-k relevant chunks retrieved → LLM answers from that context
- Ask unlimited follow-ups against the same indexed document

**Architecture (as originally built):**
```
PDF upload → extract_text.py (page-tracked text)
          → chunker.py (500-char chunks, 50-char overlap, per page)
          → embedder.py (sentence-transformers, all-MiniLM-L6-v2, 384-dim)
          → vectordb.py (FAISS IndexFlatL2 + chunk lookup)

Question   → embedder.py (same model)
          → vectordb.py (top-k similarity search)
          → generator.py (Gemini API answers from retrieved chunks only)
```

**Tech stack:**

| Stage | Tool | Why |
|---|---|---|
| PDF extraction | `pdfplumber` | Page-by-page, preserves page numbers |
| Chunking | Custom, fixed-size + overlap | Simple, predictable, per-page |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local, CPU-friendly |
| Vector store | FAISS (`IndexFlatL2`) | Built from scratch, not Chroma — to learn the mechanics |
| Generation (primary) | Gemini API (`gemini-3.5-flash`) | Free tier |
| Generation (fallback) | Groq API (`llama-3.3-70b-versatile`) | Independent quota, added in Part 3 |
| Backend | Flask + Gunicorn | Matches prior project pattern |
| Deployment | Docker + Render (free tier) | CPU-only PyTorch to keep image small |

**Key decisions:**
- Chunking is per-page → every chunk stays attributable to one page number
- 50-char overlap → ideas split at a chunk boundary still appear whole somewhere
- Local embeddings + API generation → embeddings run far more often, so keeping them free/local was the right tradeoff
- FAISS built by hand, not Chroma → deliberate, to actually learn retrieval mechanics
- CPU-only PyTorch installed explicitly → avoids the much larger default GPU build

**Known limitations at the end of Part 1:**
- Single document at a time — one global index, overwritten on every upload
- Broad/summary questions answered weakly — only a partial slice of the document reached the model
- No conversation memory — each question answered in isolation
- No retry on Gemini's occasional `503`s
- No upload validation — file type/size/corruption unchecked

All five addressed in Part 2.

---

## Part 2 — Hardening pass

Making the app behave correctly under real conditions: bad input, a flaky API, multiple users, and questions the basic retrieval model can't handle.

**1. Upload validation** — extension checked (`.pdf`), size checked against a 10MB cap directly from the byte stream, and the saved file is opened with `pdfplumber` to catch corrupt/mislabeled files before they hit the expensive pipeline steps. Failed uploads are deleted, not left behind.

**2. Gemini retry logic** — `ServerError` (5xx, transient) retries with exponential backoff (~1s, 2s, 4s); `ClientError` (4xx, permanent — bad key, bad request) fails immediately, since retrying it wastes time. Exhausted retries return a clean JSON error instead of crashing. Extended further in Part 3.

**3. Per-session isolation** — each browser gets a signed session cookie; document index and conversation history are now stored per-session instead of one shared global object. Tradeoff: session data lives in server memory only, so a restart/redeploy/spin-down wipes it — adding a database was out of scope for this pass.

**4. Broad vs. narrow question detection** — a keyword/pattern classifier flags summary-style questions ("summarize," "overview," "how many chapters," etc.); broad questions skip retrieval and get the whole document as context, narrow questions still use similarity search. Kept as pattern-matching rather than a second LLM call, deliberately — an extra API call per question doubles cost and failure surface for marginal accuracy gain at this scale.

**5. Conversation memory** — last 3 Q&A exchanges per session are sent as context on each new question, so follow-ups can reference earlier answers. Capped at 3 to keep prompt size/token cost bounded as a session grows.

**6. UI/UX redesign + feedback** — muted warm dark theme, serif title, answers styled as a left-accent margin note instead of a chat bubble, real drag-and-drop upload zone. Every answer gets 👍/👎: both log feedback, 👎 also triggers an auto-regenerate attempt (excluding the bad answer from history sent back to the model) plus an optional comment field.

**Known gap, left unfixed on purpose:** semantic search struggles with exact structural lookups (e.g. "what is question 1") — little distinctive meaning for the embedding model to match on, so it can confidently retrieve the wrong section with no error at all. Flagged for a future pass, not silently patched.

---

## Part 3 — Provider resilience (Groq fallback)

Gemini's free tier rate-limits aggressively under active use — a `429` previously failed the request immediately, by design, since retrying the *same* provider during a rate-limit window doesn't help. The gap: that's a poor experience for something meant to demo reliably.

**What changed:**
- On a `429` specifically, the app now falls back to Groq (`llama-3.3-70b-versatile`) immediately instead of failing — a different provider has an entirely independent quota, so switching is the correct response, not a retry.
- On a genuine `5xx`, the existing retry-with-backoff on Gemini still runs first (transient errors often recover in seconds); Groq is only used once those retries are exhausted, i.e. Gemini is actually down, not just briefly overloaded.
- If both providers fail, the error surfaces clearly rather than silently returning nothing.

**A real bug caught while making this change:** the original prompt-construction line mixed an inline conditional with implicit string-literal concatenation:
```python
user_prompt = (
    f"Previous conversation:\n{history_block}\n\n" if history_block else ""
    f"Context:\n{context_block}\n\n"
    f"Question: {question}"
)
```
Python concatenates adjacent string literals *before* evaluating the surrounding conditional, so whenever `history_block` was non-empty — i.e. on every question after the first in a session — the result collapsed to just the "Previous conversation" prefix, silently dropping the retrieved context and the current question from every multi-turn prompt. Single-question sessions never exposed this, which is why it passed earlier testing. Fixed with explicit `+` concatenation; verified in production across multi-turn sessions (leave policy → sick leave follow-up correctly scoped without repeating the topic).

---

## Known limitations (current)

- **Memory behavior across repeated uploads:** `VectorDB` instances are correctly replaced and garbage-collected on each new upload — verified, not a logical leak. However, FAISS and NumPy allocate their buffers through glibc's `malloc` rather than Python's own allocator, and glibc often retains freed heap pages in its own arena instead of returning them to the OS. Render's OOM killer measures RSS (actual memory held by the process), so repeated allocate/free cycles within one long-running session can push RSS upward over time even with no logical bug in the code. Mitigations (periodic worker restart, explicit `gc.collect()` + `malloc_trim()`, or capping re-uploads per session) are a documented future improvement, not yet applied, given this is a free-tier demo rather than a high-traffic deployment.
- Session data is in-memory only — wiped on restart/redeploy/spin-down.
- Single Gunicorn worker assumed — multiple workers would need shared session storage (Redis/Postgres).
- `session_store` has no expiry — every distinct session since the last restart stays resident until the process restarts. Acceptable for a low-traffic demo; would need TTL-based eviction before handling sustained concurrent users.
- Broad-question detection is keyword-based — misses unlisted phrasings; LLM-based classification considered, deferred (extra API call per question).
- Exact structural lookups (question/section numbers) can retrieve wrong content confidently.
- Gemini's `429` rate limits are now handled via Groq fallback (Part 3); Groq has its own free-tier limits (30 RPM / 1K RPD as of testing), which are generous enough for demo-level traffic but not unlimited.

---

## Running locally

```
git clone https://github.com/Kuldip-Lakhtariya/document-qa-rag.git
cd document-qa-rag
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`.env`:
```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
SECRET_KEY=any_random_string_for_local_testing
```

Run:
```
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
