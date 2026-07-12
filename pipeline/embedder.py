from typing import List, Dict
from sentence_transformers import SentenceTransformer

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Adds an 'embedding' vector to each chunk dict.

    Input:  [{"page": 1, "chunk_id": 0, "text": "..."}, ...]
    Output: [{"page": 1, "chunk_id": 0, "text": "...", "embedding": [0.02, ...]}, ...]
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = _embedding_model.encode(texts)

    for chunk, embedding_vector in zip(chunks, embeddings):
        chunk["embedding"] = embedding_vector.tolist()

    return chunks


def embed_query(query_text: str) -> List[float]:
    """
    Embeds a single user question the same way chunks were embedded.
    Must use the SAME model as embed_chunks — otherwise the question's
    vector and the chunks' vectors won't live in the same space, and
    similarity comparison becomes meaningless.
    """
    return _embedding_model.encode(query_text).tolist()