from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name: str, distance: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, vectors: List[List[float]], limit: int = 5, filters: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

    @abstractmethod
    def delete(self, vector_id: str):
        raise NotImplementedError

    @abstractmethod
    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

    @abstractmethod
    def get(self, vector_id: str):
        raise NotImplementedError

    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

