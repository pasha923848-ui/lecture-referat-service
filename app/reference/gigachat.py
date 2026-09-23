"""Minimal GigaChat (Sber) API client — an alternative to the local
llama-cpp model for writing реферат text. Cloud-based, so it needs an
internet connection and an Authorization key (from
https://developers.sber.ru/studio, "Ключ авторизации" for GigaChat API),
but is much faster than a 3B model running on CPU.

The key is stored in the app's own SQLite settings table (see app.db),
editable at runtime through the web UI (POST /settings/gigachat) — not an
env var — so a student can turn this on without rebuilding/restarting the
container.
"""
import base64
import threading
import time
import uuid

import requests

_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
# Requested explicitly (not the default "GigaChat" model) — handles
# long-form academic Russian text noticeably better.
_MODEL = "GigaChat-Pro"

# GigaChat's certificate chain is issued by the Russian Minцифры root CA,
# which isn't in most systems' trust stores — verifying against it would
# require bundling that CA cert. Disabling verification is what every
# GigaChat quick-start (including Sber's own SDK, by default) does for
# exactly this reason; the trade-off is accepted here as this only carries
# an anonymized lecture transcript, not credentials.
_VERIFY = False

_lock = threading.Lock()
_token_cache: dict = {}  # auth_key -> {"token": str, "expires_at": float}


class GigaChatError(RuntimeError):
    pass


def _get_access_token(auth_key: str) -> str:
    with _lock:
        cached = _token_cache.get(auth_key)
        if cached and time.time() < cached["expires_at"] - 30:
            return cached["token"]

        try:
            resp = requests.post(
                _TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_key}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": "GIGACHAT_API_PERS"},
                verify=_VERIFY,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise GigaChatError(f"Не удалось связаться с сервером авторизации GigaChat: {exc}") from exc

        if resp.status_code != 200:
            raise GigaChatError(
                f"GigaChat отклонил ключ авторизации (код {resp.status_code}): {resp.text[:300]}"
            )
        data = resp.json()
        token = data["access_token"]
        # expires_at is epoch milliseconds.
        _token_cache[auth_key] = {"token": token, "expires_at": data["expires_at"] / 1000}
        return token


def generate(system_prompt: str, user_prompt: str, max_tokens: int, api_key: str) -> str:
    """Same shape as app.reference.llm.generate (system/user prompt in,
    text out) so it's a drop-in replacement as generate_fn everywhere a
    реферат section is written."""
    token = _get_access_token(api_key)

    try:
        resp = requests.post(
            _CHAT_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.6,
            },
            verify=_VERIFY,
            # A ~1300-word answer is ~3500 output tokens; at GigaChat-Pro's
            # generation speed that can take well over two minutes.
            timeout=300,
        )
    except requests.RequestException as exc:
        raise GigaChatError(f"Не удалось связаться с GigaChat: {exc}") from exc

    if resp.status_code != 200:
        raise GigaChatError(f"Ошибка GigaChat API (код {resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def check_key(api_key: str) -> None:
    """Raises GigaChatError if the key doesn't work — used to validate a
    key right when it's saved through the web UI, instead of only finding
    out the next time a реферат is generated."""
    _get_access_token(api_key)
