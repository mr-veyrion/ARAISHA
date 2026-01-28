from __future__ import annotations

import os
import re
import sqlite3
import threading
from typing import Dict, Iterable, List, Optional, Tuple

from .local_config import EntityIndexConfig


ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_\-]+)\b")


class EntityIndex:
    """Lightweight entity lexicon and alias index backed by SQLite.

    Schema:
      entities(entity TEXT PRIMARY KEY)
      aliases(alias TEXT PRIMARY KEY, canonical TEXT)
      memory_entities(memory_id TEXT, entity TEXT)
    """

    def __init__(self, config: Optional[EntityIndexConfig] = None):
        self.config = config or EntityIndexConfig()
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS entities(entity TEXT PRIMARY KEY)")
            cur.execute("CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY, canonical TEXT)")
            cur.execute("CREATE TABLE IF NOT EXISTS memory_entities(memory_id TEXT, entity TEXT)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_entities_mid ON memory_entities(memory_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity)")
            self.conn.commit()

    def _extract_entities(self, text: str) -> List[str]:
        # Heuristic: Proper-case tokens + quoted phrases (to capture lowercase names)
        ents = set(ENTITY_PATTERN.findall(text))
        quoted = re.findall(r'"([^"]+)"', text)
        for q in quoted:
            q = q.strip()
            if not q:
                continue
            # Keep first token capitalized as entity key
            parts = q.split()
            if parts:
                ents.add(parts[0].strip().capitalize())
        return list(ents)

    def add_memory_entities(self, memory_id: str, text: str, additional_entities: Optional[List[str]] = None):
        ents = self._extract_entities(text)
        if additional_entities:
            ents = list(set(ents) | set(additional_entities))
        with self._lock:
            cur = self.conn.cursor()
            for e in ents:
                cur.execute("INSERT OR IGNORE INTO entities(entity) VALUES (?)", (e,))
                cur.execute("INSERT INTO memory_entities(memory_id, entity) VALUES (?, ?)", (memory_id, e))
            self.conn.commit()

    def entities_for_memory(self, memory_id: str) -> List[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT entity FROM memory_entities WHERE memory_id=?", (memory_id,))
            return [r[0] for r in cur.fetchall()]

    def memories_for_entity(self, entity: str) -> List[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT memory_id FROM memory_entities WHERE entity=?", (entity,))
            return [r[0] for r in cur.fetchall()]

    def add_alias(self, alias: str, canonical: str):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO aliases(alias, canonical) VALUES (?, ?)", (alias, canonical))
            self.conn.commit()

    def resolve(self, name: str) -> str:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT canonical FROM aliases WHERE alias=?", (name,))
            row = cur.fetchone()
            return row[0] if row else name

    def delete_by_memory_id(self, memory_id: str) -> None:
        """Delete all entity associations for a given memory_id."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM memory_entities WHERE memory_id=?", (memory_id,))
            self.conn.commit()

    def delete_all(self) -> None:
        """Delete all entries from entities, aliases, and memory_entities tables."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM memory_entities")
            cur.execute("DELETE FROM entities")
            cur.execute("DELETE FROM aliases")
            self.conn.commit()

    def memories_for_entity_case_insensitive(self, entity: str) -> List[str]:
        """Get all memory IDs associated with an entity (case-insensitive)."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT memory_id FROM memory_entities WHERE LOWER(entity)=LOWER(?)", (entity,))
            return [r[0] for r in cur.fetchall()]
