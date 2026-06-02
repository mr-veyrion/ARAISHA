from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

import numpy as np

from .local_config import LocalEmbedderConfig


class LocalEmbedding:
    def __init__(self, config: Optional[LocalEmbedderConfig] = None):
        self.config = config or LocalEmbedderConfig()
        # Encourage offline mode for HF
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Quieter transformers logging to avoid noisy tokenizer advisories
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        # Prefer local model path resolution
        path = self._resolve_model_path(self.config.model_path)
        # Use BGE-M3 via FlagEmbedding
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"FlagEmbedding (BGE-M3) not available: {e}")
        # Use fp16 to speed up on compatible hardware (CPU fallback will ignore)
        self.model = BGEM3FlagModel(path, use_fp16=True)
        # Embedding dims expected by config (dense vectors)
        self._dims = int(self.config.embedding_dims)

    def embed(self, text: str) -> list[float]:
        text = (text or "").replace("\n", " ").strip()
        if not text:
            return np.zeros(self._dims, dtype=np.float32).tolist()
        out = self.model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense: np.ndarray = out["dense_vecs"][0]
        vec = dense.astype(np.float32)
        if vec.shape[0] != self._dims:
            pad = np.zeros(self._dims, dtype=np.float32)
            pad[: min(self._dims, vec.shape[0])] = vec[: self._dims]
            return pad.tolist()
        return vec.tolist()

    def embed_with_lexical(self, texts: List[str]) -> Dict[str, Any]:
        """Batch-embed returning dense vectors and lexical weights for each input.
        Returns a dict: {"dense": List[List[float]], "lexical": List[Dict[str, float]]}
        """
        cleaned = [(t or "").replace("\n", " ").strip() for t in texts]
        # Guard against empty list
        if not cleaned:
            return {"dense": [], "lexical": []}
        out = self.model.encode(
            cleaned,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=getattr(self.config, "max_length", 1024),
        )
        dense_list: List[np.ndarray] = [v.astype(np.float32) for v in out.get("dense_vecs", [])]
        lexical_list = out.get("lexical_weights", [])
        # Normalize dims and pad/truncate as needed
        norm_dense: List[List[float]] = []
        for vec in dense_list:
            if vec.shape[0] != self._dims:
                pad = np.zeros(self._dims, dtype=np.float32)
                pad[: min(self._dims, vec.shape[0])] = vec[: self._dims]
                norm_dense.append(pad.tolist())
            else:
                norm_dense.append(vec.tolist())
        return {"dense": norm_dense, "lexical": lexical_list}

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = self.embed_with_lexical(texts)
        return res["dense"]

    def lexical_score(self, query_lexical: Dict[str, float], doc_lexical: Dict[str, float]) -> float:
        """Compute lexical matching score between query and doc (BM25-like)."""
        try:
            return float(self.model.compute_lexical_matching_score(query_lexical, doc_lexical))
        except Exception:
            return 0.0

    def lexical_scores(self, query_lexical: Dict[str, float], doc_lexicals: List[Dict[str, float]]) -> List[float]:
        scores: List[float] = []
        for d in doc_lexicals:
            scores.append(self.lexical_score(query_lexical, d))
        return scores

    def pair_scores_m3(self, pairs: List[List[str]], max_passage_length: int = 128, weights: List[float] | None = None) -> Dict[str, List[float]]:
        """Compute dense/sparse/colbert and fused scores using BGE-M3 for query-doc pairs.
        Returns dict with keys: 'dense', 'sparse', 'colbert', 'sparse+dense', 'colbert+sparse+dense'.
        """
        if not pairs:
            return {"colbert": [], "sparse": [], "dense": [], "sparse+dense": [], "colbert+sparse+dense": []}
        try:
            w = weights or [0.4, 0.2, 0.4]  # recommended in docs
            # Use the explicit argument for passage length to respect caller's config (e.g., reranker.max_passage_length)
            mpl = max_passage_length
            return self.model.compute_score(pairs, max_passage_length=mpl, weights_for_different_modes=w)
        except Exception:
            return {"colbert": [], "sparse": [], "dense": [], "sparse+dense": [], "colbert+sparse+dense": []}

    def advanced_rerank_m3(self, query: str, docs: List[str], max_passage_length: int = 128) -> List[Dict[str, float]]:
        """
        Advanced reranking using BGE-M3 with multiple fusion strategies and enhanced logic.
        Returns detailed scoring breakdown for each document.
        """
        if not docs:
            return []
            
        pairs = [[query, doc] for doc in docs]
        
        try:
            # Get all scoring modes from BGE-M3
            base_scores = self.model.compute_score(pairs, max_passage_length=max_passage_length)
            
            dense_scores = base_scores.get("dense", [])
            sparse_scores = base_scores.get("sparse", [])
            colbert_scores = base_scores.get("colbert", [])
            
            results = []
            
            for i, doc in enumerate(docs):
                if i >= len(dense_scores):
                    break
                    
                dense = float(dense_scores[i]) if i < len(dense_scores) else 0.0
                sparse = float(sparse_scores[i]) if i < len(sparse_scores) else 0.0
                colbert = float(colbert_scores[i]) if i < len(colbert_scores) else 0.0
                
                # Advanced fusion strategies
                score_breakdown = self._compute_advanced_fusion(query, doc, dense, sparse, colbert)
                
                results.append(score_breakdown)
            
            return results
            
        except Exception:
            # Fallback to simple dense scoring if BGE-M3 compute_score fails
            return [{"final_score": 0.5, "dense": 0.5, "sparse": 0.0, "colbert": 0.0, "fusion_type": "fallback"} for _ in docs]
    
    def _compute_advanced_fusion(self, query: str, doc: str, dense: float, sparse: float, colbert: float) -> Dict[str, float]:
        """
        Compute advanced fusion with multiple strategies and query-adaptive weighting.
        """
        query_lower = query.lower()
        doc_lower = doc.lower()
        
        # Query characteristics analysis
        query_words = query_lower.split()
        doc_words = doc_lower.split()
        
        # 1. Query type detection for adaptive weighting
        is_factual = any(word in query_lower for word in ['who', 'what', 'when', 'where', 'which'])
        is_conceptual = any(word in query_lower for word in ['how', 'why', 'explain', 'describe', 'about'])
        is_short_query = len(query_words) <= 3
        is_long_query = len(query_words) > 8
        
        # 2. Content overlap analysis
        word_overlap = len(set(query_words) & set(doc_words)) / max(len(query_words), 1)
        exact_matches = sum(1 for word in query_words if word in doc_lower)
        
        # 3. Adaptive weight calculation based on query characteristics
        if is_factual and word_overlap > 0.3:
            # Factual queries benefit from exact matching (sparse)
            weights = [0.3, 0.5, 0.2]  # dense, sparse, colbert
            fusion_type = "factual_sparse"
        elif is_conceptual:
            # Conceptual queries benefit from semantic understanding (dense)
            weights = [0.6, 0.2, 0.2]
            fusion_type = "conceptual_dense"
        elif is_short_query:
            # Short queries benefit from fine-grained matching (colbert)
            weights = [0.3, 0.3, 0.4]
            fusion_type = "short_colbert"
        elif is_long_query:
            # Long queries have enough context for dense matching
            weights = [0.5, 0.3, 0.2]
            fusion_type = "long_dense"
        else:
            # Balanced approach for general queries
            weights = [0.4, 0.3, 0.3]
            fusion_type = "balanced"
        
        # 4. Content-based weight adjustment
        if exact_matches > 0:
            # Boost sparse for exact matches
            weights[1] = min(0.6, weights[1] + 0.2)
            weights[0] = max(0.2, weights[0] - 0.1)
            fusion_type += "_boosted_sparse"
        
        if word_overlap < 0.1 and dense > 0.7:
            # Boost dense for semantic similarity when no word overlap
            weights[0] = min(0.7, weights[0] + 0.2)
            weights[1] = max(0.1, weights[1] - 0.1)
            fusion_type += "_boosted_dense"
        
        # 5. Multiple fusion strategies
        linear_fusion = weights[0] * dense + weights[1] * sparse + weights[2] * colbert
        
        # Geometric mean for balanced scoring
        geometric_fusion = (dense ** weights[0]) * (sparse ** weights[1]) * (colbert ** weights[2])
        
        # Harmonic mean for conservative scoring
        harmonic_components = []
        if dense > 0: harmonic_components.append(weights[0] / dense)
        if sparse > 0: harmonic_components.append(weights[1] / sparse)
        if colbert > 0: harmonic_components.append(weights[2] / colbert)
        harmonic_fusion = len(harmonic_components) / sum(harmonic_components) if harmonic_components else 0.0
        
        # Max fusion for best-of-breed
        max_fusion = max(dense * weights[0], sparse * weights[1], colbert * weights[2])
        
        # 6. Final adaptive fusion
        if word_overlap > 0.5:
            # High overlap: prefer linear or max
            final_score = 0.7 * linear_fusion + 0.3 * max_fusion
        elif dense > 0.8 or sparse > 0.8 or colbert > 0.8:
            # High confidence in one mode: boost it
            final_score = 0.6 * linear_fusion + 0.4 * max_fusion
        else:
            # Balanced case: use geometric mean for smooth scoring
            final_score = 0.5 * linear_fusion + 0.3 * geometric_fusion + 0.2 * harmonic_fusion
        
        # 7. Quality boost for high-confidence matches
        if max(dense, sparse, colbert) > 0.85:
            final_score = min(1.0, final_score * 1.1)
        
        # 8. Length normalization bonus
        if len(doc_words) > 10 and final_score > 0.6:
            # Slight bonus for longer, relevant documents
            final_score = min(1.0, final_score * 1.05)
        
        return {
            "final_score": float(final_score),
            "dense": float(dense),
            "sparse": float(sparse), 
            "colbert": float(colbert),
            "linear_fusion": float(linear_fusion),
            "geometric_fusion": float(geometric_fusion),
            "harmonic_fusion": float(harmonic_fusion),
            "max_fusion": float(max_fusion),
            "weights": weights,
            "word_overlap": float(word_overlap),
            "exact_matches": int(exact_matches),
            "fusion_type": fusion_type
        }

    def graph_enhanced_rerank_m3(self, query: str, memory_records: List[Dict], graph_contexts: List[Dict], max_passage_length: int = 128) -> List[Dict[str, float]]:
        """
        Graph-enhanced BGE-M3 reranking that combines memory text with graph structure information.
        Uses src--relation--dst format to create richer context for better retrieval accuracy.
        """
        if not memory_records:
            return []
        
        try:
            # Build graph-enhanced text representations
            enhanced_pairs = []
            context_info = []
            
            for i, (memory_record, graph_ctx) in enumerate(zip(memory_records, graph_contexts)):
                memory_text = memory_record.get("memory", "").strip()
                memory_id = memory_record.get("id", "")
                
                if not memory_text:
                    continue
                
                # Extract graph context elements
                entities = graph_ctx.get("memory_entities", [])
                direct_rels = graph_ctx.get("direct_relationships", [])
                entity_overlap = graph_ctx.get("entity_overlap", 0)
                graph_confidence = graph_ctx.get("graph_confidence", 0.0)
                traversal_strength = graph_ctx.get("traversal_strength", 0.0)
                
                # Create graph-enhanced memory representation
                graph_context_text = ""
                if entities:
                    graph_context_text += f" [ENTITIES: {', '.join(entities[:5])}]"
                if direct_rels:
                    # Format relationships for better BGE-M3 understanding
                    rel_context = []
                    for rel in direct_rels[:3]:  # Top 3 most relevant
                        if "--" in rel and "-->" in rel:
                            # Parse "SRC --REL--> DST" format
                            try:
                                parts = rel.split("--")
                                if len(parts) >= 3:
                                    src = parts[0].strip()
                                    rel_part = parts[1].strip()
                                    dst = parts[2].split("-->")[1].strip() if "-->" in parts[2] else parts[2].strip()
                                    rel_context.append(f"{src} {rel_part.replace('_', ' ').lower()} {dst}")
                            except:
                                rel_context.append(rel.replace("--", " ").replace("-->", " "))
                    if rel_context:
                        graph_context_text += f" [RELATIONSHIPS: {'; '.join(rel_context)}]"
                
                # Create enhanced text for BGE-M3
                enhanced_memory_text = f"{memory_text}{graph_context_text}"
                enhanced_pairs.append([query, enhanced_memory_text])
                
                # Store context info for post-processing
                context_info.append({
                    "entity_overlap": entity_overlap,
                    "graph_confidence": graph_confidence,
                    "traversal_strength": traversal_strength,
                    "has_entities": len(entities) > 0,
                    "has_relationships": len(direct_rels) > 0,
                    "original_text": memory_text
                })
            
            if not enhanced_pairs:
                return []
            
            # Get BGE-M3 scores with graph-enhanced context
            base_scores = self.model.compute_score(enhanced_pairs, max_passage_length=max_passage_length)
            
            dense_scores = base_scores.get("dense", [])
            sparse_scores = base_scores.get("sparse", [])
            colbert_scores = base_scores.get("colbert", [])
            
            # Apply graph-aware enhancements to each scoring mode
            enhanced_results = []
            
            for i, ctx_info in enumerate(context_info):
                if i >= len(dense_scores):
                    break
                
                dense = float(dense_scores[i]) if i < len(dense_scores) else 0.0
                sparse = float(sparse_scores[i]) if i < len(sparse_scores) else 0.0
                colbert = float(colbert_scores[i]) if i < len(colbert_scores) else 0.0
                
                # Graph-aware score boosting
                entity_boost = min(0.3, ctx_info["entity_overlap"] * 0.15)  # Boost for entity matches
                confidence_boost = min(0.2, ctx_info["graph_confidence"] * 0.1)  # Boost for graph confidence
                traversal_boost = min(0.25, ctx_info["traversal_strength"] * 0.12)  # Boost for traversal paths
                
                # Apply different boosts to different scoring modes
                enhanced_dense = dense + entity_boost + (confidence_boost * 0.5)
                enhanced_sparse = sparse + (entity_boost * 1.5) + confidence_boost  # Sparse benefits more from entity matches
                enhanced_colbert = colbert + traversal_boost + (entity_boost * 0.8)  # ColBERT benefits from fine-grained matching
                
                # Adaptive weighting based on graph context
                if ctx_info["entity_overlap"] > 2:
                    # High entity overlap: boost sparse and colbert
                    weights = [0.3, 0.4, 0.3]
                    fusion_type = "entity_focused"
                elif ctx_info["has_relationships"] and ctx_info["graph_confidence"] > 0.6:
                    # Strong relationships: boost colbert for fine-grained matching
                    weights = [0.3, 0.3, 0.4]
                    fusion_type = "relationship_focused"
                elif ctx_info["traversal_strength"] > 0.7:
                    # Strong traversal paths: balanced with slight dense bias
                    weights = [0.4, 0.3, 0.3]
                    fusion_type = "traversal_focused"
                else:
                    # Default balanced approach
                    weights = [0.35, 0.35, 0.3]
                    fusion_type = "balanced_graph"
                
                # Compute enhanced fusion
                linear_fusion = weights[0] * enhanced_dense + weights[1] * enhanced_sparse + weights[2] * enhanced_colbert
                
                # Confidence-based final adjustment
                confidence_factor = 1.0 + (ctx_info["graph_confidence"] * 0.1)
                final_score = linear_fusion * confidence_factor
                
                # Quality boost for high-confidence graph matches
                if (ctx_info["entity_overlap"] > 1 and ctx_info["graph_confidence"] > 0.8):
                    final_score = min(1.0, final_score * 1.1)
                
                enhanced_results.append({
                    "final_score": float(final_score),
                    "dense": float(enhanced_dense),
                    "sparse": float(enhanced_sparse),
                    "colbert": float(enhanced_colbert),
                    "original_dense": float(dense),
                    "original_sparse": float(sparse),
                    "original_colbert": float(colbert),
                    "entity_boost": float(entity_boost),
                    "confidence_boost": float(confidence_boost),
                    "traversal_boost": float(traversal_boost),
                    "fusion_type": fusion_type,
                    "graph_enhanced": True,
                    "entity_overlap": ctx_info["entity_overlap"],
                    "graph_confidence": ctx_info["graph_confidence"],
                    "weights": weights
                })
            
            return enhanced_results
            
        except Exception as e:
            # Fallback to basic scoring if graph enhancement fails
            print(f"[graph_enhanced_rerank_warning] {e}, falling back to basic scoring")
            return self.advanced_rerank_m3(query, [r.get("memory", "") for r in memory_records], max_passage_length)

    def _resolve_model_path(self, base_path: str) -> str:
        base_path = os.path.abspath(base_path)
        if os.path.isdir(base_path):
            direct_files = [
                os.path.join(base_path, "config.json"),
                os.path.join(base_path, "modules.json"),
                os.path.join(base_path, "pytorch_model.bin"),
            ]
            if any(os.path.exists(p) for p in direct_files):
                return base_path
            snap = os.path.join(base_path, "snapshots")
            if os.path.isdir(snap):
                for name in os.listdir(snap):
                    cand = os.path.join(snap, name)
                    if os.path.isdir(cand):
                        files = [
                            os.path.join(cand, "config.json"),
                            os.path.join(cand, "modules.json"),
                            os.path.join(cand, "pytorch_model.bin"),
                        ]
                        if any(os.path.exists(p) for p in files):
                            return cand
        return base_path

