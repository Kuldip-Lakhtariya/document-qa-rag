from google import genai
from google.genai import types
from typing import List, Dict

SYSTEM_INSTRUCTION = (
    "You are a document Q&A assistant. Answer using the provided context. "
    "If the exact answer isn't stated directly, but can be reasonably "
    "inferred from the context (e.g. counting visible section headers), "
    "do so and say it's an inference. Only say the information is not "
    "available if there is genuinely nothing in the context related to "
    "the question — don't refuse just because the exact phrasing isn't there."
)

_client = genai.Client()


def generate_answer(retrieved_chunks: List[Dict[str, object]], question: str) -> str:
    context_block = "\n\n".join(
        f"[Page {chunk['page']}]: {chunk['text']}"
        for chunk in retrieved_chunks
    )

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    response = _client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1024,  # raised from 500 — leaves room for thinking + full answer
            thinking_config=types.ThinkingConfig(thinking_level="low")  # minimal reasoning needed for direct Q&A
        )
    )

    return response.text
