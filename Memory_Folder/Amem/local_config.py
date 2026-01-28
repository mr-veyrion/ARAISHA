from __future__ import annotations

import os
from pydantic import BaseModel, Field
from typing import Any, Dict
import yaml


def _default_root() -> str:
    return os.path.join(os.getcwd(), "models")


class LocalLLMConfig(BaseModel):
    model_path: str = Field(
        default_factory=lambda: _default_root(),
        description="Path to local HF model for chat/instruction LLM.",
    )
    backend: str = Field(
        default="auto",
        description="LLM backend selection: 'auto' | 'gguf' (auto when .gguf present)",
    )
    device: str = Field(default="auto")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=0.9)
    use_8bit: bool = Field(default=False)
    use_4bit: bool = Field(default=False)


class LocalEmbedderConfig(BaseModel):
    model_path: str = Field(
        default_factory=lambda: os.path.join(_default_root(), "bge-m3"),
        description="Path to local BGE-M3 embedding model (FlagEmbedding).",
    )
    embedding_dims: int = Field(default=1024)
    max_length: int = Field(default=1024, description="Max sequence length for BGE-M3 encode (lower = faster)")


class EmotionAnalyzerConfig(BaseModel):
    model_path: str = Field(
        default_factory=lambda: os.path.join(_default_root(), "roberta-base-go_emotions"),
        description="Path to local RoBERTa emotion analysis model.",
    )
    device: str = Field(default="auto", description="Device for emotion analysis: 'auto', 'cpu', or 'cuda'")
    max_length: int = Field(default=512, description="Max sequence length for emotion analysis")
    batch_size: int = Field(default=32, description="Batch size for emotion analysis")
    confidence_threshold: float = Field(default=0.1, description="Minimum confidence threshold for emotion detection")


class LocalVectorStoreConfig(BaseModel):
    collection_name: str = Field(default="Amem")
    path: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "local_data", "faiss"))
    distance_strategy: str = Field(default="cosine")  # cosine | euclidean | inner_product
    index_type: str = Field(default="HNSW")  # Flat | HNSW | IVF
    normalize: bool = Field(default=True)
    embedding_model_dims: int = Field(default=1024)


class LocalGraphConfig(BaseModel):
    db_path: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "local_data", "graph.db"))


class EntityIndexConfig(BaseModel):
    db_path: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "local_data", "entity_index.db"))


class LocalNodeVectorConfig(BaseModel):
    path: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "local_data", "node_vectors"))
    collection_name: str = Field(default="node_vectors")
    distance_strategy: str = Field(default="cosine")
    index_type: str = Field(default="HNSW")
    embedding_model_dims: int = Field(default=1024)


class ScoringWeights(BaseModel):
    w_similarity: float = Field(default=0.6)
    w_graph: float = Field(default=0.2)
    w_recency: float = Field(default=0.15)
    w_evidence: float = Field(default=0.05)
    w_lexical: float = Field(default=0.2)


class RetrievalConfig(BaseModel):
    initial_top_k: int = Field(default=50)  # Increased for better recall
    final_top_k: int = Field(default=20)  # Increased for better accuracy
    merge_similarity_threshold: float = Field(default=0.82)  # cosine
    update_similarity_threshold: float = Field(default=0.9)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    min_answer_score: float = Field(default=0.25)
    allow_no_answer: bool = Field(default=True)
    debug: bool = Field(default=False)
    traverse_debug: bool = Field(default=False)


class RerankerConfig(BaseModel):
    enabled: bool = Field(default=True)
    top_k: int = Field(default=25)  # Increased for better accuracy
    max_passage_length: int = Field(default=256, description="Max passage length for BGE-M3 advanced fusion")
    advanced_fusion: bool = Field(default=True, description="Use advanced BGE-M3 fusion strategies")
    debug_scores: bool = Field(default=False, description="Show detailed scoring breakdown")


class EntityMergingConfig(BaseModel):
    """Configuration for entity and relationship similarity merging."""
    entity_similarity_threshold: float = Field(default=0.85, description="Cosine similarity threshold for merging entities")
    relationship_similarity_threshold: float = Field(default=0.8, description="Similarity threshold for merging relationships")
    enable_entity_merging: bool = Field(default=False, description="Enable automatic entity merging during storage (disabled by default for performance)")
    enable_relationship_merging: bool = Field(default=False, description="Enable automatic relationship merging during storage (disabled by default for performance)")
    max_merge_candidates: int = Field(default=5, description="Maximum candidates to consider for merging")


class OfflineMemoryConfig(BaseModel):
    llm: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    embedder: LocalEmbedderConfig = Field(default_factory=LocalEmbedderConfig)
    emotion_analyzer: EmotionAnalyzerConfig = Field(default_factory=EmotionAnalyzerConfig)
    vector_store: LocalVectorStoreConfig = Field(default_factory=LocalVectorStoreConfig)
    graph: LocalGraphConfig = Field(default_factory=LocalGraphConfig)
    entity_index: EntityIndexConfig = Field(default_factory=EntityIndexConfig)
    node_vectors: LocalNodeVectorConfig = Field(default_factory=LocalNodeVectorConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    entity_merging: EntityMergingConfig = Field(default_factory=EntityMergingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    history_db_path: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "local_data", "history.db")
    )

    @classmethod
    def from_file(cls, path: str = "configs/offline.yaml") -> "OfflineMemoryConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        base_dir = os.path.dirname(os.path.abspath(path))

        def _resolve(p: str | None) -> str | None:
            if not p:
                return p
            if os.path.isabs(p):
                return p
            return os.path.abspath(os.path.join(base_dir, "..", p))

        # Backward/typo compatibility: allow 'ebedder' key
        if "ebedder" in data and "embedder" not in data:
            data["embedder"] = data["ebedder"]

        llm_cfg = data.get("llm", {})
        emb_cfg = data.get("embedder", {})
        emotion_cfg = data.get("emotion_analyzer", {})
        vs_cfg = data.get("vector_store", {})
        graph_cfg = data.get("graph", {})
        ent_idx_cfg = data.get("entity_index", {})
        node_vec_cfg = data.get("node_vectors", {})
        ret_cfg = data.get("retrieval", {})
        rerank_cfg = data.get("reranker", {})

        # Resolve paths
        if "model_path" in llm_cfg:
            llm_cfg["model_path"] = _resolve(llm_cfg.get("model_path"))
        if "model_path" in emb_cfg:
            emb_cfg["model_path"] = _resolve(emb_cfg.get("model_path"))
        if "model_path" in emotion_cfg:
            emotion_cfg["model_path"] = _resolve(emotion_cfg.get("model_path"))
        if "path" in vs_cfg:
            vs_cfg["path"] = _resolve(vs_cfg.get("path"))
        if "db_path" in graph_cfg:
            graph_cfg["db_path"] = _resolve(graph_cfg.get("db_path"))
        if "db_path" in ent_idx_cfg:
            ent_idx_cfg["db_path"] = _resolve(ent_idx_cfg.get("db_path"))
        if "path" in node_vec_cfg:
            node_vec_cfg["path"] = _resolve(node_vec_cfg.get("path"))
        if "model_path" in rerank_cfg:
            rerank_cfg["model_path"] = _resolve(rerank_cfg.get("model_path"))

        return cls(
            llm=LocalLLMConfig(**llm_cfg) if llm_cfg else LocalLLMConfig(),
            embedder=LocalEmbedderConfig(**emb_cfg) if emb_cfg else LocalEmbedderConfig(),
            emotion_analyzer=EmotionAnalyzerConfig(**emotion_cfg) if emotion_cfg else EmotionAnalyzerConfig(),
            vector_store=LocalVectorStoreConfig(**vs_cfg) if vs_cfg else LocalVectorStoreConfig(),
            graph=LocalGraphConfig(**graph_cfg) if graph_cfg else LocalGraphConfig(),
            entity_index=EntityIndexConfig(**ent_idx_cfg) if ent_idx_cfg else EntityIndexConfig(),
            node_vectors=LocalNodeVectorConfig(**node_vec_cfg) if node_vec_cfg else LocalNodeVectorConfig(),
            retrieval=RetrievalConfig(**ret_cfg) if ret_cfg else RetrievalConfig(),
            reranker=RerankerConfig(**rerank_cfg) if rerank_cfg else RerankerConfig(),
        )

