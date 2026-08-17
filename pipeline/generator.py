from google import genai
from google.genai import types, errors
from groq import Groq
from typing import List, Dict, Tuple, Optional
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

_gemini_client = genai.Client()
_groq_client = Groq()  # reads GROQ_API_KEY from env — same pattern as genai.Client()

MAX_RETRIES = 3           # 1 initial attempt + 3 retries = 4 total tries on Gemini
BASE_DELAY_SECONDS = 1.0  # doubles each retry: ~1s, ~2s, ~4s

GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"


def _build_user_prompt(
    context_block: str,
    question: str,
    history_block: str,
) -> str:
    """
    Builds the full prompt sent to whichever provider handles the request.

    NOTE: previously this used implicit string-literal concatenation next to
    a ternary, e.g.:
        f"A" if history_block else "" f"B" f"C"
    Adjacent string literals concatenate BEFORE the ternary is evaluated, so
    that version silently collapsed to just f"A" whenever history_block was
    non-empty — dropping context_block and question entirely on every
    multi-turn question. Explicit `+` concatenation below avoids that trap.
    """
    history_prefix = f"Previous conversation:\n{history_block}\n\n" if history_block else ""
    return (
        history_prefix
        + f"Context:\n{context_block}\n\n"
        + f"Question: {question}"
    )


def _call_groq(user_prompt: str) -> str:
    """Fallback path — used when Gemini is rate-limited (429) or its retries
    on transient 5xx errors are exhausted (i.e. genuinely down, not just a
    bad millisecond)."""
    response = _groq_client.chat.completions.create(
        model=GROQ_FALLBACK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def generate_answer(
    retrieved_chunks: List[Dict[str, object]],
    question: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str,str]:
    conversation_history = conversation_history or []
    history_block = "\n\n".join(
        f"Q: {past_q}\nA: {past_a}" for past_q, past_a in conversation_history
    )
    context_block = "\n\n".join(
        f"[Page {chunk['page']}]: {chunk['text']}"
        for chunk in retrieved_chunks
    )
    user_prompt = _build_user_prompt(context_block, question, history_block)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _gemini_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=user_prompt,
                config=config,
            )
            return response.text,"Gemini"

        except errors.ServerError as gemini_error:
            # 5xx (503 "overloaded" is the common one) — transient, Gemini's
            # side. Give the existing backoff a chance to recover first;
            # only fall back to a different provider once retries are
            # genuinely exhausted.
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            try:
                return _call_groq(user_prompt),"groq" 
            except Exception as groq_error:
                raise RuntimeError(
                    f"Both providers failed. Gemini: {gemini_error}. Groq: {groq_error}"
                ) from groq_error

        except errors.ClientError as gemini_error:
            status_code = getattr(gemini_error, "code", None)
            if status_code == 429:
                try:
                    return _call_groq(user_prompt),"Groq"
                except Exception as groq_error:
                    raise RuntimeError(
                        f"Both providers failed. Gemini: {gemini_error}. Groq: {groq_error}"
                    ) from groq_error
            raise
