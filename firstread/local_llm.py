"""
Minimal client for a LOCAL Ollama server (localhost only).

Used by scripts/local_grade.py for local-only preliminary grading: student
writing is scrubbed by utils/pii_gate and sent to a model running on this
machine — nothing leaves the localhost boundary. Deliberately kept outside
utils/model_catalog.py, which is Anthropic-specific.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:26b"

# Ollama's own default context window is small (often 4-8k) regardless of what
# the model supports — a long paper + rubric would be silently truncated from
# the FRONT of the prompt. Always set num_ctx explicitly.
DEFAULT_NUM_CTX = 16384


class LocalLLMError(RuntimeError):
    pass


def _post(host: str, path: str, payload: dict, timeout: int) -> dict:
    if not host.startswith(("http://localhost", "http://127.0.0.1")):
        # This module exists so student work stays on this machine. Refuse
        # anything else rather than becoming an accidental emission path.
        raise LocalLLMError(f"local_llm refuses non-localhost host: {host}")
    req = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except OSError as e:
        raise LocalLLMError(
            f"Ollama not reachable at {host} ({e}). Start it with: ollama serve"
        ) from e


def chat_json(prompt: str, *, model: str = DEFAULT_MODEL,
              host: str = DEFAULT_HOST, num_ctx: int = DEFAULT_NUM_CTX,
              temperature: float = 1.0, timeout: int = 900,
              schema: dict | None = None, retries: int = 1,
              think: bool = False, images: list | None = None,
              seed: int = 0) -> dict:
    """Send one prompt, get a parsed JSON object back.

    Pass a JSON ``schema`` whenever the result shape is known: with the bare
    ``format: "json"`` mode gemma has been observed to emit a degenerate
    ``{}``, while schema-constrained decoding forces the required keys.

    Sampling defaults follow the gemma model card (temp 1.0, top_k 64,
    top_p 0.95): LOWER temperatures under grammar constraints send gemma
    into repetition loops that never close the JSON. Degeneration is
    stochastic, so a parse failure is retried on a fresh seed.

    ``think`` is off by default: Ollama enables gemma4's thinking mode
    unless told otherwise, and on short classification prompts the model
    has been observed to spend the whole ``num_predict`` budget thinking
    (~15k chars, 75 s) and return EMPTY content. Thinking tokens count
    toward ``num_predict``.

    ``images`` (raw PNG/JPEG bytes, or base64 strings) ride along with the
    prompt for a vision-capable model — a scanned page to transcribe.
    ``seed`` lets a caller re-ask when a parseable answer is still unusable.
    """
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = [i if isinstance(i, str)
                             else base64.b64encode(i).decode("ascii")
                             for i in images]
    last_err = None
    for attempt in range(retries + 1):
        data = _post(host, "/api/chat", {
            "model": model,
            "messages": [message],
            "format": schema or "json",
            "stream": False,
            "think": bool(think),
            "options": {"num_ctx": num_ctx, "temperature": temperature,
                        "top_k": 64, "top_p": 0.95,
                        "num_predict": 4000, "seed": seed + attempt},
        }, timeout)
        content = ((data.get("message") or {}).get("content") or "").strip()
        try:
            if not content:
                raise LocalLLMError(f"empty response from {model}")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
                raise LocalLLMError(
                    f"model returned non-JSON: {content[:400]}")
        except (LocalLLMError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < retries:
                log.warning("local model output unparseable (attempt %d), "
                            "retrying: %s", attempt + 1, str(e)[:200])
    raise LocalLLMError(str(last_err))


def list_models(host: str = DEFAULT_HOST, timeout: int = 10) -> list[str]:
    req = urllib.request.Request(host.rstrip("/") + "/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except OSError as e:
        raise LocalLLMError(
            f"Ollama not reachable at {host} ({e}). Start it with: ollama serve"
        ) from e


def model_capabilities(model: str, host: str = DEFAULT_HOST,
                       timeout: int = 10) -> list[str]:
    """What the model can do, per Ollama (``vision``, ``thinking``, …).
    Empty when Ollama doesn't know the model or reports nothing for it."""
    try:
        return list(_post(host, "/api/show", {"model": model}, timeout)
                    .get("capabilities") or [])
    except LocalLLMError:
        return []
