from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .dsl import Query, MatchPattern, WherePredicate
from .vector_store import LocalVectorStore
from .graph_store import GraphStore, Relation
from .local_embedding import LocalEmbedding
from .algorithms import shortest_path, path_to_relations, dijkstra_shortest_path
from .node_vectors import NodeVectorIndex


class Planner:
    def __init__(self, graph: GraphStore, vectors: LocalVectorStore, embed: LocalEmbedding, node_vectors: Optional[NodeVectorIndex] = None):
        self.graph = graph
        self.vectors = vectors
        self.embed = embed
        self.node_vectors = node_vectors

    def execute(self, q: Query) -> Dict[str, Any]:
        # Very simple plan:
        # 1) If SIMILAR(var, ...) present → vector prefilter to node names
        # 2) Expand via graph by label/name constraints
        # 3) Apply prop_eq filters
        # 4) Return node/edge rows up to limit
        similar_names: List[str] = []
        for pred in q.where:
            if pred.kind == 'similar':
                _, query_str, topk, thr = pred.args
                vec = self.embed.embed(query_str)
                # prefer node vectors if available for faster graph prefiltering
                if self.node_vectors is not None:
                    hits = self.node_vectors.search(vec, limit=topk)
                else:
                    hits = self.vectors.search(vec, limit=topk)
                # Use payload.memory strings as candidate names
                for vid, score, payload in hits:
                    if score >= thr:
                        name = (payload or {}).get('name') or (payload or {}).get('memory')
                        if name:
                            similar_names.append(name)

        # Try shortestPath if variable-length and endpoints provided via props
        rels: List[Relation] = []
        if q.match.rel.max_hops and q.match.rel.max_hops > 1:
            left_name = q.match.left.props.get('name')
            right_name = q.match.right.props.get('name')
            if left_name and right_name:
                # choose weighted if hops large
                if q.match.rel.max_hops > 4:
                    path = dijkstra_shortest_path(self.graph, left_name, right_name)
                else:
                    path = shortest_path(self.graph, left_name, right_name, max_depth=q.match.rel.max_hops)
                rels = path_to_relations(self.graph, path)
        if not rels:
            # Fallback edge scan - more efficient query
            rels = self.graph.query_edges(
                relationship=q.match.rel.types[0] if q.match.rel.types else None,
                source_name=None,
                destination_name=None,
                limit=min(q.limit * 10, 500),  # Reduced from 50x to 10x for performance
            )
        rows = []
        for r in rels:
            # Vector prefilter on names
            if similar_names and not any(n in (r.source or '') or n in (r.destination or '') for n in similar_names):
                continue
            # Relationship type filter
            if q.match.rel.types is not None and r.relationship.upper() not in q.match.rel.types:
                continue
            # Label constraints
            if q.match.left.label and not self.graph.node_has_label(r.source, q.match.left.label):
                continue
            if q.match.right.label and not self.graph.node_has_label(r.destination, q.match.right.label):
                continue
            # Property equals on node-level props
            failed = False
            for pred in q.where:
                if pred.kind == 'prop_eq':
                    var, key, value = pred.args
                    if var == q.match.left.var:
                        pv = self.graph.get_node_prop_by_name(r.source, key)
                        if pv != value:
                            failed = True
                            break
                    elif var == q.match.right.var:
                        pv = self.graph.get_node_prop_by_name(r.destination, key)
                        if pv != value:
                            failed = True
                            break
            if failed:
                continue
            rows.append({
                'source': r.source,
                'relationship': r.relationship,
                'destination': r.destination,
                'memory_id': r.memory_id,
            })
            if len(rows) >= q.limit:
                break
        return {'rows': rows}

