"""
server_llm.py — API-backed LLM client (drop-in for LocalLLM)

Purpose
-------
- Provide an API-based generation backend that matches the interface used by the
  existing system (generate_messages / generate_messages_stream), so it can be
  plugged in anywhere the local LLM is used (e.g., normal chat and relationship
  extraction prompts).

Notes
-----
- Prompt structure is unchanged; callers pass the same messages structure
  (list of {"role": "system"|"user"|"assistant", "content": str}).
- This client uses NVIDIA's integrate API via the OpenAI SDK-compatible client.
- API key resolution: pass api_key explicitly or set env var NVIDIA_API_KEY.
  No silent fallbacks; a missing key raises an error.
"""

from __future__ import annotations

from typing import List, Dict, Any, Iterator, Optional
import os
from dotenv import load_dotenv
load_dotenv()


class ServerLLM:
    """API-backed LLM client with the same interface as the local LLM.

    Methods:
      - generate_messages(messages, max_new_tokens, temperature, top_p) -> str
      - generate_messages_stream(messages, max_new_tokens, temperature, top_p) -> Iterator[str]

    The prompt structure and message format are unchanged from the existing system.
    """

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        deployment_name: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: Optional[str] = None,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        default_max_tokens: int = 1000,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package is required for ServerLLM. pip install openai") from e

        # Allow override of base url from env for flexibility
        base_url = os.getenv("OPENAI_BASE_URL", base_url)

        # Prefer provider-appropriate key. If Azure-style endpoint is used,
        # prefer OPENAI_API_KEY/AZURE_OPENAI_API_KEY over NVIDIA_API_KEY.
        is_azure = ("azure.com" in (base_url or "")) or ("openai.azure" in (base_url or ""))
        if is_azure:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
        else:
            resolved_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "NVIDIA_API_KEY or OPENAI_API_KEY not set and no api_key provided for ServerLLM"
            )

        # Construct client
        self._client = OpenAI(base_url=base_url, api_key=resolved_key)
        # For Azure, the 'model' field expects the deployment name. Prefer deployment_name when set.
        self._model = deployment_name or model_name
        self._default_temperature = float(default_temperature)
        self._default_top_p = float(default_top_p)
        self._default_max_tokens = int(default_max_tokens)

        # Expose a backend label to align with existing checks (no code changes elsewhere)
        self.backend: str = "server"

    # -----------------------------
    # Public API
    # -----------------------------
    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body={"reasoning_budget":16384,"chat_template_kwargs":{"enable_thinking":False}}
    ) -> str:
        """Return a full string completion for the given chat messages.

        This preserves the exact message structure and system prompts passed by callers
        (e.g., relationship extraction and normal chat), so prompt behavior is unchanged.
        """
        payload = self._normalize_messages(messages)

        # For Azure, 'model' should be the deployment name; ensure base_url ends with /openai/v1/
        _base_url_str = str(getattr(self._client, "base_url", "") or "")
        if 'azure.com' in _base_url_str and not _base_url_str.rstrip('/').endswith('/openai/v1'):
            # Best-effort guidance; not fatal
            pass
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=self._pick(temperature, self._default_temperature),
            top_p=self._pick(top_p, self._default_top_p),
            max_tokens=self._pick(max_new_tokens, self._default_max_tokens),
            extra_body={"reasoning_budget":16384,"chat_template_kwargs":{"enable_thinking":False}},
            stream=False,
        )

        # Concatenate all parts of the first choice content (if segmented)
        try:
            choice = resp.choices[0]
            # New OpenAI SDK: message is an object with 'content'
            content = getattr(choice.message, "content", None)
            if content is None:
                # Fallback: assemble from deltas if provided (rare on non-stream)
                return ""
            return content
        except Exception as e:
            raise RuntimeError(f"ServerLLM.generate_messages failed: {e}") from e

    def generate_messages_stream(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body={"reasoning_budget":16384,"chat_template_kwargs":{"enable_thinking":False}}
    ) -> Iterator[str]:
        """Yield incremental text chunks for the given chat messages (streaming)."""
        payload = self._normalize_messages(messages)

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=self._pick(temperature, self._default_temperature),
            top_p=self._pick(top_p, self._default_top_p),
            max_tokens=self._pick(max_new_tokens, self._default_max_tokens),
            stream=True,
        )

        try:
            for chunk in stream:
                # Optionally surface reasoning content if present
                delta_obj = None
                try:
                    delta_obj = chunk.choices[0].delta
                except Exception:
                    delta_obj = None
                if delta_obj is None:
                    continue
                reasoning = getattr(delta_obj, "reasoning_content", None)
                if reasoning:
                    yield str(reasoning)
                content_piece = getattr(delta_obj, "content", None)
                if content_piece:
                    yield str(content_piece)
        except Exception as e:
            raise RuntimeError(f"ServerLLM.generate_messages_stream failed: {e}") from e

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _pick(val: Optional[float | int], default_val: float | int) -> float | int:
        return default_val if val is None else val

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Ensure messages match the OpenAI-compatible schema.

        Accepts the existing system/user/assistant message dicts unchanged; coerces types.
        """
        if not isinstance(messages, list):
            raise TypeError("messages must be a list of {role, content} dicts")
        out: List[Dict[str, str]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            out.append({"role": role, "content": content})
        return out


# -----------------------------
# Minimal usage example (script)
# -----------------------------
if __name__ == "__main__":
    # Example: stream a simple completion. Set NVIDIA_API_KEY in env or pass api_key.
    base_url = os.getenv("OPENAI_BASE_URL", "https://abhik-ma8bxst0-eastus2.cognitiveservices.azure.com/openai/v1/")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
    llm = ServerLLM(api_key=api_key, base_url=base_url, model_name=os.getenv("OPENAI_MODEL", "gpt-4o"), deployment_name=os.getenv("OPENAI_DEPLOYMENT", "gpt-4o"))
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in a cheerful tone."},
    ]
    for chunk in llm.generate_messages_stream(msgs, max_new_tokens=200):
        print(chunk, end="")
    print()


