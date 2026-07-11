from typing import List, Dict


def chunk_text(
    extracted_pages: List[Dict[str, object]],
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, object]]:
    """
    Splits page-tracked text into overlapping chunks.

    Each page's text is chunked independently (never combined across
    pages), so every chunk stays attributable to exactly one page number.

    Returns a list like:
    [{"page": 1, "chunk_id": 0, "text": "..."}, ...]
    """
    all_chunks: List[Dict[str, object]] = []

    for page_data in extracted_pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        start = 0
        chunk_id = 0

        while start < len(page_text):
            end = start + chunk_size
            chunk_text_piece = page_text[start:end]

            all_chunks.append({
                "page": page_number,
                "chunk_id": chunk_id,
                "text": chunk_text_piece
            })

            start += (chunk_size - overlap) # overlap characters.
            chunk_id += 1

    return all_chunks
