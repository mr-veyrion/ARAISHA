from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import uuid
import difflib

from .local_config import LocalGraphConfig


@dataclass
class Relation:
    source: str
    relationship: str
    destination: str
    created_at: str
    updated_at: str
    weight: float = 1.0
    memory_id: str | None = None


class GraphStore:
    """SQLite-backed property graph store (edges only) for simplicity and scale.

    Schema:
      nodes(id TEXT PRIMARY KEY, name TEXT, label TEXT, props TEXT, cluster_id TEXT, created_at TEXT, updated_at TEXT)
      edges(source TEXT, relationship TEXT, destination TEXT, memory_id TEXT, source_id TEXT, destination_id TEXT, created_at TEXT, updated_at TEXT, weight REAL)
      indexes on nodes(name), nodes(label), edges(source), edges(destination), edges(relationship), edges(memory_id), edges(source_id), edges(destination_id)
    """

    def __init__(self, config: Optional[LocalGraphConfig] = None):
        self.config = config or LocalGraphConfig()
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _init(self):
        with self._lock:
            cur = self.conn.cursor()
            # nodes table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                  id TEXT PRIMARY KEY,
                  name TEXT,
                  label TEXT,
                  props TEXT,
                  cluster_id TEXT,
                  created_at TEXT,
                  updated_at TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label)")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_nodes_name_label ON nodes(name, label)")
            # multi-labels table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS node_labels (
                  node_id TEXT,
                  label TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nlabels_node ON node_labels(node_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nlabels_label ON node_labels(label)")
            # node properties (KV)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS node_props (
                  node_id TEXT,
                  key TEXT,
                  value TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nprops_node ON node_props(node_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nprops_key ON node_props(key)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                  source TEXT,
                  relationship TEXT,
                  destination TEXT,
                  memory_id TEXT,
                  source_id TEXT,
                  destination_id TEXT,
                  created_at TEXT,
                  updated_at TEXT,
                  weight REAL
                )
                """
            )
            # Migration: add columns if missing
            cur.execute("PRAGMA table_info(edges)")
            cols = {row[1] for row in cur.fetchall()}
            if "memory_id" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN memory_id TEXT")
            if "source_id" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN source_id TEXT")
            if "destination_id" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN destination_id TEXT")
            # edge properties (KV)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_props (
                  source TEXT,
                  relationship TEXT,
                  destination TEXT,
                  key TEXT,
                  value TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_eprops_trip ON edge_props(source,relationship,destination)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_eprops_key ON edge_props(key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_dest ON edges(destination)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relationship)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_mid ON edges(memory_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_sid ON edges(source_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_did ON edges(destination_id)")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_edge_trip ON edges(source, relationship, destination)")
            self.conn.commit()

    def add(self, rel: Relation):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO edges(source, relationship, destination, memory_id, source_id, destination_id, created_at, updated_at, weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rel.source,
                    rel.relationship,
                    rel.destination,
                    rel.memory_id,
                    None,
                    None,
                    rel.created_at,
                    rel.updated_at,
                    rel.weight,
                ),
            )
            self.conn.commit()

    def upsert(self, rel: Relation):
        existing = self.find(rel.source, rel.relationship, rel.destination)
        if existing:
            # increment weight to accumulate evidence
            new_w = float((existing.weight or 1.0)) + float(rel.weight or 1.0)
            self.update(rel.source, rel.relationship, rel.destination, rel.updated_at, new_w, rel.memory_id)
        else:
            self.add(rel)

    def update(self, source: str, relationship: str, destination: str, updated_at: str, weight: float = 1.0, memory_id: str | None = None):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE edges SET updated_at=?, weight=?, memory_id=COALESCE(?, memory_id) WHERE source=? AND relationship=? AND destination=?",
                (updated_at, weight, memory_id, source, relationship, destination),
            )
            self.conn.commit()

    def delete(self, source: str, relationship: str, destination: str):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "DELETE FROM edges WHERE source=? AND relationship=? AND destination=?",
                (source, relationship, destination),
            )
            self.conn.commit()

    def delete_by_memory_id(self, memory_id: str):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM edges WHERE memory_id=?", (memory_id,))
            self.conn.commit()

    def delete_all(self):
        """Delete all nodes and edges from the graph."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM edges")
            cur.execute("DELETE FROM nodes")
            self.conn.commit()

    def delete_by_entity_case_insensitive(self, entity: str) -> int:
        """Delete all edges containing the entity (case-insensitive). Returns number of deleted edges."""
        with self._lock:
            cur = self.conn.cursor()
            # Delete edges where entity is source or destination (case-insensitive)
            cur.execute(
                "DELETE FROM edges WHERE LOWER(source)=LOWER(?) OR LOWER(destination)=LOWER(?)",
                (entity, entity),
            )
            deleted_count = cur.rowcount
            self.conn.commit()
            return deleted_count


    # ---------- transactions ----------
    def transaction(self):
        store = self
        class _Tx:
            def __enter__(self_inner):
                store.conn.execute("BEGIN")
                return store
            def __exit__(self_inner, exc_type, exc, tb):
                if exc_type is not None:
                    store.conn.execute("ROLLBACK")
                else:
                    store.conn.execute("COMMIT")
        return _Tx()

    # -------- nodes API --------
    def upsert_node(self, node_id: str, name: str, label: str, props_json: str, created_at: str, updated_at: str):
        with self._lock:
            cur = self.conn.cursor()
            # Try update first
            cur.execute("UPDATE nodes SET name=?, label=?, props=?, updated_at=? WHERE id=?", (name, label, props_json, updated_at, node_id))
            if cur.rowcount == 0:
                cur.execute(
                    "INSERT INTO nodes(id, name, label, props, cluster_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (node_id, name, label, props_json, None, created_at, updated_at),
                )
            self.conn.commit()

    def get_node_by_name(self, name: str) -> Optional[Tuple[str, str, str, str, str, str, str]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, name, label, props, cluster_id, created_at, updated_at FROM nodes WHERE name=?", (name,))
            return cur.fetchone()

    def node_id_by_name(self, name: str) -> Optional[str]:
        row = self.get_node_by_name(name)
        return row[0] if row else None

    def node_has_label(self, name: str, label: str) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT 1 FROM nodes WHERE name=? AND label=? LIMIT 1", (name, label))
            if cur.fetchone():
                return True
            cur.execute(
                "SELECT 1 FROM node_labels nl JOIN nodes n ON nl.node_id = n.id WHERE n.name=? AND nl.label=? LIMIT 1",
                (name, label),
            )
            return cur.fetchone() is not None

    def get_node_prop_by_name(self, name: str, key: str) -> Optional[str]:
        node_id = self.node_id_by_name(name)
        if not node_id:
            return None
        return self.get_node_prop(node_id, key)
    
    def get_node_prop(self, node_id: str, key: str) -> Optional[str]:
        """Get a property value for a node by node_id."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT value FROM node_props WHERE node_id=? AND key=? LIMIT 1", (node_id, key))
            row = cur.fetchone()
            return row[0] if row else None
    
    def set_node_prop(self, node_id: str, key: str, value: str) -> None:
        """Set a property value for a node."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT 1 FROM node_props WHERE node_id=? AND key=?", (node_id, key))
            exists = cur.fetchone() is not None
            if exists:
                cur.execute("UPDATE node_props SET value=? WHERE node_id=? AND key=?", (value, node_id, key))
            else:
                cur.execute("INSERT INTO node_props(node_id, key, value) VALUES(?,?,?)", (node_id, key, value))
            self.conn.commit()

    def get_node_by_id(self, node_id: str) -> Optional[Tuple[str, str, str, str, str, str, str]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, name, label, props, cluster_id, created_at, updated_at FROM nodes WHERE id=?", (node_id,))
            return cur.fetchone()

    def delete_node(self, node_id: str):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM edges WHERE source_id=? OR destination_id=?", (node_id, node_id))
            cur.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            self.conn.commit()

    def set_edge_node_ids(self, source: str, destination: str, source_id: Optional[str], destination_id: Optional[str]):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE edges SET source_id=COALESCE(?, source_id), destination_id=COALESCE(?, destination_id) WHERE source=? AND destination=?",
                (source_id, destination_id, source, destination),
            )
            self.conn.commit()

    def get_edge_weight(self, a: str, b: str) -> float:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT weight FROM edges WHERE source=? AND destination=? LIMIT 1",
                (a, b),
            )
            row = cur.fetchone()
            if row:
                return float(row[0] or 1.0)
            cur.execute(
                "SELECT weight FROM edges WHERE source=? AND destination=? LIMIT 1",
                (b, a),
            )
            row = cur.fetchone()
            return float(row[0] or 1.0) if row else 1.0

    def edges_for_node(self, node_id: str, limit: int = 100) -> List[Relation]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE source_id=? OR destination_id=? LIMIT ?",
                (node_id, node_id, limit),
            )
            rows = cur.fetchall()
        return [Relation(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]

    def recompute_clusters(self) -> None:
        # Simple connected components by node_id using edges source_id/destination_id
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM nodes")
            node_ids = [r[0] for r in cur.fetchall()]
            parent = {nid: nid for nid in node_ids}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            cur.execute("SELECT source_id, destination_id FROM edges WHERE source_id IS NOT NULL AND destination_id IS NOT NULL")
            for sid, did in cur.fetchall():
                if sid in parent and did in parent:
                    union(sid, did)

            # Assign cluster ids
            cluster_id_map: Dict[str, str] = {}
            for nid in node_ids:
                root = find(nid)
                if root not in cluster_id_map:
                    cluster_id_map[root] = root  # simple cluster id = root
                cid = cluster_id_map[root]
                cur.execute("UPDATE nodes SET cluster_id=? WHERE id=?", (cid, nid))
            self.conn.commit()

    # -------- convenience: ensure node and upsert edge by names --------
    def ensure_node(self, name: str, label: str, props_json: str, created_at: str, updated_at: str) -> str:
        existing = self.get_node_by_name(name)
        if existing and existing[2] == label:  # label match
            node_id = existing[0]
            # update timestamp/props if changed
            self.upsert_node(node_id, name, label, props_json, existing[5] or created_at, updated_at)
            return node_id
        node_id = f"{label}::{name}"
        self.upsert_node(node_id, name, label, props_json, created_at, updated_at)
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO node_labels(node_id,label) VALUES(?,?)", (node_id, label))
            self.conn.commit()
        return node_id

    def upsert_edge_by_names(
        self,
        source_name: str,
        source_label: str,
        relationship: str,
        destination_name: str,
        destination_label: str,
        memory_id: Optional[str],
        created_at: str,
        updated_at: str,
        weight: float = 1.0,
    ) -> None:
        import json as _json
        sid = self.ensure_node(source_name, source_label, props_json=_json.dumps({}), created_at=created_at, updated_at=updated_at)
        did = self.ensure_node(destination_name, destination_label, props_json=_json.dumps({}), created_at=created_at, updated_at=updated_at)
        with self._lock:
            cur = self.conn.cursor()
            # Insert or update edge row
            cur.execute(
                "SELECT 1 FROM edges WHERE source=? AND relationship=? AND destination=?",
                (source_name, relationship, destination_name),
            )
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(
                    "INSERT INTO edges(source, relationship, destination, memory_id, source_id, destination_id, created_at, updated_at, weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (source_name, relationship, destination_name, memory_id, sid, did, created_at, updated_at, weight),
                )
            else:
                cur.execute(
                    "UPDATE edges SET updated_at=?, memory_id=COALESCE(?, memory_id), weight=? , source_id=COALESCE(?, source_id), destination_id=COALESCE(?, destination_id) WHERE source=? AND relationship=? AND destination=?",
                    (updated_at, memory_id, weight, sid, did, source_name, relationship, destination_name),
                )
            self.conn.commit()

    def delete_edge_by_names(self, source_name: str, relationship: str, destination_name: str) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "DELETE FROM edges WHERE source=? AND relationship=? AND destination=?",
                (source_name, relationship, destination_name),
            )
            self.conn.commit()

    def query_edges(
        self,
        relationship: Optional[str] = None,
        source_name: Optional[str] = None,
        destination_name: Optional[str] = None,
        limit: Optional[int] = 200,
    ) -> List[Relation]:
        with self._lock:
            cur = self.conn.cursor()
            conds = []
            params: List[str] = []
            if relationship:
                conds.append("relationship=?")
                params.append(relationship)
            if source_name:
                conds.append("source=?")
                params.append(source_name)
            if destination_name:
                conds.append("destination=?")
                params.append(destination_name)
            where = (" WHERE " + " AND ".join(conds)) if conds else ""
            sql = f"SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges{where}"
            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [Relation(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]

    # ---------- edge property helpers ----------
    def set_edge_prop(self, source: str, relationship: str, destination: str, key: str, value: str) -> None:
        """Upsert a key/value property for a specific edge."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT 1 FROM edge_props WHERE source=? AND relationship=? AND destination=? AND key=?",
                (source, relationship, destination, key),
            )
            exists = cur.fetchone() is not None
            if exists:
                cur.execute(
                    "UPDATE edge_props SET value=? WHERE source=? AND relationship=? AND destination=? AND key=?",
                    (value, source, relationship, destination, key),
                )
            else:
                cur.execute(
                    "INSERT INTO edge_props(source, relationship, destination, key, value) VALUES(?,?,?,?,?)",
                    (source, relationship, destination, key, value),
                )
            self.conn.commit()

    def get_edge_prop(self, source: str, relationship: str, destination: str, key: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT value FROM edge_props WHERE source=? AND relationship=? AND destination=? AND key=? LIMIT 1",
                (source, relationship, destination, key),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_edge_props(self, source: str, relationship: str, destination: str) -> Dict[str, str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT key, value FROM edge_props WHERE source=? AND relationship=? AND destination=?",
                (source, relationship, destination),
            )
            rows = cur.fetchall()
        return {k: v for (k, v) in rows}

    def find(self, source: str, relationship: str, destination: str) -> Optional[Relation]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT source, relationship, destination, created_at, updated_at, weight FROM edges WHERE source=? AND relationship=? AND destination=?",
                (source, relationship, destination),
            )
            row = cur.fetchone()
            if row:
                return Relation(*row)
            return None

    def neighbors(self, entity: str, direction: str = "both", relationship: Optional[str] = None, limit: int = 100) -> List[Relation]:
        with self._lock:
            cur = self.conn.cursor()
            if direction == "out":
                if relationship:
                    cur.execute(
                        "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE source=? AND relationship=? LIMIT ?",
                        (entity, relationship, limit),
                    )
                else:
                    cur.execute(
                        "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE source=? LIMIT ?",
                        (entity, limit),
                    )
            elif direction == "in":
                if relationship:
                    cur.execute(
                        "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE destination=? AND relationship=? LIMIT ?",
                        (entity, relationship, limit),
                    )
                else:
                    cur.execute(
                        "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE destination=? LIMIT ?",
                        (entity, limit),
                    )
            else:
                if relationship:
                    cur.execute(
                        "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE (source=? OR destination=?) AND relationship=? LIMIT ?",
                        (entity, entity, relationship, limit),
                    )
                else:
                    cur.execute(
                        "SELECT source, relationship, destination, created_at, updated_at, weight, memory_id FROM edges WHERE source=? OR destination=? LIMIT ?",
                        (entity, entity, limit),
                    )
            rows = cur.fetchall()
        return [Relation(*row) for row in rows]

    def k_hop(self, entity: str, k: int = 2, limit_per_hop: int = 50) -> List[Relation]:
        visited = set([entity])
        frontier = [entity]
        output: List[Relation] = []
        for _ in range(k):
            next_frontier = []
            for e in frontier:
                rels = self.neighbors(e, direction="both", limit=limit_per_hop)
                for r in rels:
                    output.append(r)
                    other = r.destination if r.source == e else r.source
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
            frontier = next_frontier
        return output

    # ---------- query methods ----------
    
    def query_nodes(self, limit: Optional[int] = None):
        """Get all nodes from the graph."""
        with self._lock:
            cur = self.conn.cursor()
            if limit:
                cur.execute("SELECT id, name, label, props, cluster_id, created_at, updated_at FROM nodes LIMIT ?", (limit,))
            else:
                cur.execute("SELECT id, name, label, props, cluster_id, created_at, updated_at FROM nodes")
            rows = cur.fetchall()
        
        # Return simple objects with attributes
        nodes = []
        for row in rows:
            node = type('Node', (), {})()
            node.node_id = row[0]
            node.name = row[1] 
            node.label = row[2]
            node.props = row[3]
            node.cluster_id = row[4]
            node.created_at = row[5]
            node.updated_at = row[6]
            nodes.append(node)
        return nodes

    # unified query_edges above handles both filtered and full-list cases

    # ---------- entity/relationship similarity and merging ----------
    
    def find_similar_entities(self, entity_name: str, similarity_threshold: float = 0.85, max_candidates: int = 5) -> List[Tuple[str, float]]:
        """Find entities similar to the given entity using string similarity."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT name FROM nodes WHERE name != ?", (entity_name,))
            all_entities = [row[0] for row in cur.fetchall()]
        
        similar_entities = []
        entity_lower = entity_name.lower()
        
        for candidate in all_entities:
            candidate_lower = candidate.lower()
            
            # Calculate similarity using multiple methods
            # 1. Exact case-insensitive match
            if entity_lower == candidate_lower:
                similar_entities.append((candidate, 1.0))
                continue
            
            # 2. Substring match (one contains the other)
            if entity_lower in candidate_lower or candidate_lower in entity_lower:
                ratio = min(len(entity_lower), len(candidate_lower)) / max(len(entity_lower), len(candidate_lower))
                if ratio >= similarity_threshold:
                    similar_entities.append((candidate, ratio))
                    continue
            
            # 3. Sequence similarity (handles typos, partial matches)
            seq_ratio = difflib.SequenceMatcher(None, entity_lower, candidate_lower).ratio()
            if seq_ratio >= similarity_threshold:
                similar_entities.append((candidate, seq_ratio))
            
            # 4. Word-level similarity for multi-word entities
            entity_words = set(entity_lower.split())
            candidate_words = set(candidate_lower.split())
            if entity_words and candidate_words:
                word_similarity = len(entity_words & candidate_words) / len(entity_words | candidate_words)
                if word_similarity >= similarity_threshold:
                    combined_score = (seq_ratio + word_similarity) / 2
                    if combined_score >= similarity_threshold:
                        similar_entities.append((candidate, combined_score))
        
        # Sort by similarity score (descending) and limit results
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return similar_entities[:max_candidates]

    def find_similar_relationships(self, relationship: str, similarity_threshold: float = 0.8, max_candidates: int = 5) -> List[Tuple[str, float]]:
        """Find relationships similar to the given relationship."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT relationship FROM edges WHERE relationship != ?", (relationship,))
            all_relationships = [row[0] for row in cur.fetchall()]
        
        similar_relationships = []
        rel_lower = relationship.lower()
        
        for candidate in all_relationships:
            candidate_lower = candidate.lower()
            
            # Calculate similarity
            if rel_lower == candidate_lower:
                similar_relationships.append((candidate, 1.0))
                continue
            
            # Sequence similarity
            seq_ratio = difflib.SequenceMatcher(None, rel_lower, candidate_lower).ratio()
            if seq_ratio >= similarity_threshold:
                similar_relationships.append((candidate, seq_ratio))
            
            # Word-level similarity (for multi-word relationships)
            rel_words = set(rel_lower.split('_'))
            candidate_words = set(candidate_lower.split('_'))
            if rel_words and candidate_words:
                word_similarity = len(rel_words & candidate_words) / len(rel_words | candidate_words)
                if word_similarity >= similarity_threshold:
                    combined_score = (seq_ratio + word_similarity) / 2
                    if combined_score >= similarity_threshold:
                        similar_relationships.append((candidate, combined_score))
        
        similar_relationships.sort(key=lambda x: x[1], reverse=True)
        return similar_relationships[:max_candidates]

    def merge_entities(self, source_entity: str, target_entity: str) -> Dict[str, int]:
        """Merge source_entity into target_entity, updating all references."""
        merge_stats = {"nodes_merged": 0, "edges_updated": 0}
        
        with self._lock:
            cur = self.conn.cursor()
            
            # Start transaction
            cur.execute("BEGIN")
            
            try:
                # 1. Update all edges where source_entity appears as source
                cur.execute(
                    "UPDATE edges SET source = ? WHERE source = ?",
                    (target_entity, source_entity)
                )
                merge_stats["edges_updated"] += cur.rowcount
                
                # 2. Update all edges where source_entity appears as destination
                cur.execute(
                    "UPDATE edges SET destination = ? WHERE destination = ?",
                    (target_entity, source_entity)
                )
                merge_stats["edges_updated"] += cur.rowcount
                
                # 3. Update source_id and destination_id references
                source_node = self.get_node_by_name(source_entity)
                target_node = self.get_node_by_name(target_entity)
                
                if source_node and target_node:
                    source_id, target_id = source_node[0], target_node[0]
                    
                    cur.execute(
                        "UPDATE edges SET source_id = ? WHERE source_id = ?",
                        (target_id, source_id)
                    )
                    cur.execute(
                        "UPDATE edges SET destination_id = ? WHERE destination_id = ?",
                        (target_id, source_id)
                    )
                
                # 4. Remove duplicate edges that might have been created
                cur.execute("""
                    DELETE FROM edges WHERE rowid NOT IN (
                        SELECT MIN(rowid) FROM edges 
                        GROUP BY source, relationship, destination
                    )
                """)
                
                # 5. Delete the source node
                if source_node:
                    cur.execute("DELETE FROM nodes WHERE name = ?", (source_entity,))
                    cur.execute("DELETE FROM node_labels WHERE node_id = ?", (source_node[0],))
                    cur.execute("DELETE FROM node_props WHERE node_id = ?", (source_node[0],))
                    merge_stats["nodes_merged"] = 1
                
                # 6. Update node_vectors if they exist (handle in memory.py)
                
                cur.execute("COMMIT")
                
            except Exception as e:
                cur.execute("ROLLBACK")
                raise e
        
        return merge_stats

    def merge_relationships(self, source_rel: str, target_rel: str) -> Dict[str, int]:
        """Merge source relationship type into target relationship type."""
        merge_stats = {"relationships_merged": 0}
        
        with self._lock:
            cur = self.conn.cursor()
            
            cur.execute("BEGIN")
            
            try:
                # Update all edges with source_rel to use target_rel
                cur.execute(
                    "UPDATE edges SET relationship = ? WHERE relationship = ?",
                    (target_rel, source_rel)
                )
                merge_stats["relationships_merged"] = cur.rowcount
                
                # Remove duplicate edges that might have been created
                cur.execute("""
                    DELETE FROM edges WHERE rowid NOT IN (
                        SELECT MIN(rowid) FROM edges 
                        GROUP BY source, relationship, destination
                    )
                """)
                
                cur.execute("COMMIT")
                
            except Exception as e:
                cur.execute("ROLLBACK")
                raise e
        
        return merge_stats

