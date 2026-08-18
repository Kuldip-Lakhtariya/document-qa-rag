from google import genai
from google.genai import types

_gemini_client = genai.Client()

RELEVANCE_SYSTEM_INSTRUCTION = (
    "You judge whether retrieved document excerpts contain enough "
    "information to answer a question. Respond with exactly one word: "
    "'yes' if the excerpts are sufficient, 'no' if they are not."
)


def is_relevant(question: str, context_chunks: list) -> bool:
    """Cheap, low-token LLM check — not a full answer, just a yes/no judgment."""
    context_block = "\n\n".join(
        f"[Page {chunk['page']}]: {chunk['text']}" for chunk in context_chunks
    )
    prompt = f"Question: {question}\n\nRetrieved excerpts:\n{context_block}"

    config = types.GenerateContentConfig(
        system_instruction=RELEVANCE_SYSTEM_INSTRUCTION,
        temperature=0.0,          # deterministic judgment, not creative
        max_output_tokens=5,      
    )
    response = _gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=config,
    )
    return response.text.strip().lower().startswith("yes")
