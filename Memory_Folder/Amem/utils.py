from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def md5_hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def merge_texts(text_a: str, text_b: str) -> str:
    if text_a == text_b:
        return text_a
    # Simple merge: prefer longer with unique tokens, can be improved
    tokens_a = set((text_a or "").split())
    tokens_b = set((text_b or "").split())
    merged = list(tokens_a | tokens_b)
    return " ".join(merged)

