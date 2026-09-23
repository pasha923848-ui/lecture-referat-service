"""Thin wrapper around a local GGUF instruct model (llama-cpp-python),
loaded once and reused — same lazy-singleton pattern as
app.transcription.get_model for Whisper. Fully offline: the model file is
baked into the Docker image at build time (see Dockerfile), so nothing is
downloaded at runtime.
"""
from functools import lru_cache

from app.config import LLM_CONTEXT_SIZE, LLM_MAX_TOKENS, LLM_MODEL_PATH, LLM_THREADS


@lru_cache(maxsize=1)
def get_model():
    from llama_cpp import Llama

    # Loaded from the literal path the Dockerfile copied the weights to at
    # build time — NOT Llama.from_pretrained(repo_id=..., filename=...),
    # which resolves `filename` via the HF Hub API on every call (even for
    # an already-cached file) and hard-fails under HF_HUB_OFFLINE=1.
    return Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_CONTEXT_SIZE,
        n_threads=LLM_THREADS,
        verbose=False,
    )


def generate(system_prompt: str, user_prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """Run one chat-style completion and return the assistant's reply text."""
    model = get_model()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.6,
    )
    return response["choices"][0]["message"]["content"].strip()
