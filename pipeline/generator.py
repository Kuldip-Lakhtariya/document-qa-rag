from google import genai
from google.genai import types, errors
from typing import List, Dict
import time
import random

SYSTEM_INSTRUCTION = (
    "You are a document Q&A assistant. Answer using the provided context. "
    "If the exact answer isn't stated directly, but can be reasonably "
    "inferred from the context (e.g. counting visible section headers), "
    "do so and say it's an inference. Only say the information is not "
    "available if there is genuinely nothing in the context related to "
    "the question — don't refuse just because the exact phrasing isn't there."
)

_client = genai.Client()

MAX_RETRIES = 3           # 1 initial attempt + 3 retries = 4 total tries
BASE_DELAY_SECONDS = 1.0  # doubles each retry: ~1s, ~2s, ~4s


def generate_answer(retrieved_chunks: List[Dict[str, object]], question: str) -> str:
    context_block = "\n\n".join(
        f"[Page {chunk['page']}]: {chunk['text']}"
        for chunk in retrieved_chunks
    )
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(thinking_level="low")
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _client.models.generate_content(
                model="gemini-3.5-flash",
                contents=user_prompt,
                config=config,
            )
            return response.text

        except errors.ServerError:
            # 5xx (503 "overloaded" is the common one) — transient, Gemini's
            # side, not ours.
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            raise  # retries exhausted — bubbles up to app.py's except Exception

        except errors.ClientError:
            # 4xx — bad API key, malformed request, hard quota block. Retrying
            # identical input changes nothing, so fail immediately instead of
            # burning ~7 seconds of backoff on a request that can never succeed.
            raise
