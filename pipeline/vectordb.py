import faiss
import numpy as np
from typing import List, Dict, Tuple


class VectorDB:
    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatL2(embedding_dim)

        self.chunk_lookup: List[Dict[str, object]] = []

    def add_chunks(self, chunks: List[Dict[str, object]]) -> None:
        """
        Adds embedded chunks to the FAISS index, keeping chunk_lookup
        in the exact same order so positions stay aligned.
        """
        vectors = np.array(
            [chunk["embedding"] for chunk in chunks],
            dtype="float32"  # FAISS requires float32, not Python's default float64
        )
        self.index.add(vectors)
        self.chunk_lookup.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, object]]:
        """
        Finds the top_k most similar chunks to a query embedding.
        Returns the actual chunk dicts, not raw FAISS positions.
        """
        query_vector = np.array([query_embedding], dtype="float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for position in indices[0]:
            results.append(self.chunk_lookup[position])

        return results