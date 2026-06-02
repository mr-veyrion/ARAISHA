from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .vector_store import LocalVectorStore
from .local_config import LocalNodeVectorConfig


class NodeVectorIndex:
    def __init__(self, config: Optional[LocalNodeVectorConfig] = None):
        self.config = config or LocalNodeVectorConfig()
        self.store = LocalVectorStore(
            config=type("_cfg", (), {
                "collection_name": self.config.collection_name,
                "path": self.config.path,
                "distance_strategy": self.config.distance_strategy,
                "index_type": self.config.index_type,
                "normalize": True,
                "embedding_model_dims": self.config.embedding_model_dims,
            })()
        )

    def upsert(self, node_id: str, vector: List[float], payload: Dict[str, Any]):
        # delete if exists then add
        self.store.delete(node_id)
        self.store.add([vector], [payload], [node_id])

    def search(self, vector: List[float], limit: int = 10):
        return self.store.search(vector, limit=limit)

    def delete(self, node_id: str):
        self.store.delete(node_id)

    def items(self):
        return self.store.items()
    
    def get(self, node_id: str):
        """Get a node vector record by ID."""
        try:
            # Search for exact match by ID
            for vid, payload in self.store.items():
                if vid == node_id:
                    return payload
            return None
        except Exception:
            return None

