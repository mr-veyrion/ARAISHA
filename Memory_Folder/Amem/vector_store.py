from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss

from .local_config import LocalVectorStoreConfig


logger = logging.getLogger(__name__)


class VectorRecord:
    def __init__(self, id: str, payload: Dict[str, Any]):
        self.id = id
        self.payload = payload


class LocalVectorStore:
    """FAISS vector store with scalable index types and metadata filtering."""

    def __init__(self, config: Optional[LocalVectorStoreConfig] = None):
        self.config = config or LocalVectorStoreConfig()
        os.makedirs(self.config.path, exist_ok=True)
        self.index = None
        self.docstore: Dict[str, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self._init_index()
        self._load_if_exists()

    def _init_index(self):
        d = self.config.embedding_model_dims
        metric = faiss.METRIC_INNER_PRODUCT if self.config.distance_strategy in ("inner_product", "cosine") else faiss.METRIC_L2
        if self.config.index_type.upper() == "FLAT":
            self.index = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        elif self.config.index_type.upper() == "HNSW":
            self.index = faiss.IndexHNSWFlat(d, 32, metric)
            if self.config.distance_strategy == "cosine" or self.config.normalize:
                self.index.hnsw.efSearch = 64
        elif self.config.index_type.upper() == "IVF":
            quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, 1024, metric)
            self.index.nprobe = 16
        else:
            self.index = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)

    def _paths(self) -> Tuple[str, str]:
        index_path = os.path.join(self.config.path, f"{self.config.collection_name}.faiss")
        doc_path = os.path.join(self.config.path, f"{self.config.collection_name}.pkl")
        return index_path, doc_path

    def _load_if_exists(self):
        index_path, doc_path = self._paths()
        if os.path.exists(index_path) and os.path.exists(doc_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(doc_path, "rb") as f:
                    payload = pickle.load(f)
                    self.docstore = payload.get("docstore", {})
                    self.id_to_idx = payload.get("id_to_idx", {})
                    self.idx_to_id = payload.get("idx_to_id", {})
                logger.info("Loaded FAISS index and docstore from disk")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")

    def _save(self):
        index_path, doc_path = self._paths()
        try:
            faiss.write_index(self.index, index_path)
            with open(doc_path, "wb") as f:
                pickle.dump({"docstore": self.docstore, "id_to_idx": self.id_to_idx, "idx_to_id": self.idx_to_id}, f)
        except Exception as e:
            logger.warning(f"Failed to save FAISS index: {e}")

    def _maybe_normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.config.distance_strategy == "cosine" or self.config.normalize:
            faiss.normalize_L2(arr)
        return arr

    def add(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        if ids is None:
            raise ValueError("ids are required")
        payloads = payloads or [{} for _ in vectors]
        if len(vectors) != len(payloads) or len(vectors) != len(ids):
            raise ValueError("vectors, payloads, and ids must be same length")
        np_vecs = self._maybe_normalize(np.array(vectors, dtype=np.float32))
        if isinstance(self.index, faiss.IndexIVF):
            if not self.index.is_trained:
                self.index.train(np_vecs)
        self.index.add(np_vecs)
        start = len(self.idx_to_id)
        for i, (vid, pl) in enumerate(zip(ids, payloads)):
            self.idx_to_id[start + i] = vid
            self.id_to_idx[vid] = start + i
            self.docstore[vid] = pl.copy()
        self._save()

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None):
        if vector_id not in self.docstore:
            return
        # Get the current payload or use the provided one
        current_payload = self.docstore.get(vector_id, {})
        if payload is not None:
            current_payload = payload.copy()
            self.docstore[vector_id] = current_payload
        if vector is not None:
            # Store payload before delete
            payload_to_add = current_payload
            # Rebuild via delete+add for simplicity
            self.delete(vector_id)
            self.add([vector], [payload_to_add], [vector_id])
        else:
            self._save()

    def delete(self, vector_id: str):
        # Remove mappings and payload defensively to avoid KeyError/stale state
        idx = self.id_to_idx.pop(vector_id, None)
        if idx is not None:
            self.idx_to_id.pop(idx, None)
        # Remove payload if present
        self.docstore.pop(vector_id, None)
        # Note: FAISS does not support hard deletion for all index types; we keep the index as-is.
        # Stale entries are ignored because idx_to_id lookup will miss and be skipped in search.
        self._save()
        # For heavy churn, callers can invoke reset_index() to rebuild tightly.

    def get(self, vector_id: str) -> Optional[VectorRecord]:
        payload = self.docstore.get(vector_id)
        if payload is None:
            return None
        return VectorRecord(vector_id, payload)

    def list(self, limit: int = 100) -> List[VectorRecord]:
        out: List[VectorRecord] = []
        for vid, pl in list(self.docstore.items())[:limit]:
            out.append(VectorRecord(vid, pl.copy()))
        return out

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        return list(self.docstore.items())

    def search(self, query_vector: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        q = self._maybe_normalize(np.array(query_vector, dtype=np.float32).reshape(1, -1))
        # Search with buffer for filtering - get more candidates than needed for reranking
        k = max(limit * 2, 20)  # Get 2x limit to allow for filtering
        scores, idxs = self.index.search(q, k)
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            vid = self.idx_to_id.get(int(idx))
            if vid is None:
                continue
            payload = self.docstore.get(vid)
            if payload is None:
                continue
            if filters and not self._apply_filters(payload, filters):
                continue
            results.append((vid, float(score), payload.copy()))
            if len(results) >= limit:
                break
        return results

    def _apply_filters(self, payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Supports top-level and nested 'metadata' filters."""
        meta = payload.get("metadata") if isinstance(payload, dict) else None
        for k, v in (filters or {}).items():
            # Check top-level first, then metadata
            if isinstance(v, list):
                pv = payload.get(k)
                if pv is None and isinstance(meta, dict):
                    pv = meta.get(k)
                if pv not in v:
                    return False
            else:
                pv = payload.get(k)
                if pv is None and isinstance(meta, dict):
                    pv = meta.get(k)
                if pv != v:
                    return False
        return True

    def reset_index(self):
        """Recreate the FAISS index and mappings; keeps docstore intact."""
        self.index = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self._init_index()
        self._save()
    
    def rebuild_from_docstore(self):
        """Rebuild FAISS index from existing docstore - useful after many deletes."""
        if not self.docstore:
            return
        
        # Reset index structure
        self.reset_index()
        
        # Re-add all vectors from docstore (requires vectors to be stored in payload)
        vectors = []
        payloads = []
        ids = []
        
        for vid, payload in self.docstore.items():
            # If vector is stored in payload, rebuild
            vector = payload.get('vector')
            if vector:
                vectors.append(vector)
                payloads.append({k: v for k, v in payload.items() if k != 'vector'})
                ids.append(vid)
        
        if vectors:
            self.add(vectors, payloads, ids)

