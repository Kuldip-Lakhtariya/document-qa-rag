import google.generativeai as genai
import os
from typing import List, Dict

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_INSTRUCTION = (
    "You are a document Q&A assistant. Only answer using the provided "
    "context below. If the answer is not contained in the context, say "
    "so explicitly — do not use outside knowledge or guess."
)

_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_INSTRUCTION
)


def generate_answer(retrieved_chunks: List[Dict[str, object]], question: str) -> str:
    """
    Takes top-k retrieved chunks + the user's question, and asks Gemini
    to answer using only that context.
    """
    # Build a context block that includes page numbers, so the model
    # CAN cite them in its answer if asked to.
    context_block = "\n\n".join(
        f"[Page {chunk['page']}]: {chunk['text']}"
        for chunk in retrieved_chunks
    )

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    response = _model.generate_content(
        user_prompt,
        generation_config={
            "temperature": 0.2,   # low — factual, not creative
            "max_output_tokens": 500
        }
    )

    return response.text