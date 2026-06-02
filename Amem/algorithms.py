from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .graph_store import GraphStore, Relation


def shortest_path(graph: GraphStore, start: str, goal: str, max_depth: int = 6) -> List[str]:
    """Unweighted BFS shortest path by node names. Returns list of node names including endpoints.
    If no path found within max_depth, returns empty list.
    """
    if start == goal:
        return [start]
    visited = set([start])
    q = deque([(start, [start])])
    depth = {start: 0}
    while q:
        node, path = q.popleft()
        if depth[node] >= max_depth:
            continue
        for rel in graph.neighbors(node, direction="both", relationship=None, limit=1000):
            nxt = rel.destination if rel.source == node else rel.source
            if nxt in visited:
                continue
            visited.add(nxt)
            new_path = path + [nxt]
            if nxt == goal:
                return new_path
            depth[nxt] = depth[node] + 1
            q.append((nxt, new_path))
    return []


def path_to_relations(graph: GraphStore, path: List[str]) -> List[Relation]:
    """Given a node-name path, map it back to any existing relations along the path order.
    Chooses the first matching edge between each consecutive pair.
    """
    if len(path) < 2:
        return []
    rels: List[Relation] = []
    for a, b in zip(path[:-1], path[1:]):
        # Query either direction; pick first
        candidates = graph.query_edges(source_name=a, destination_name=b, limit=1)
        if not candidates:
            candidates = graph.query_edges(source_name=b, destination_name=a, limit=1)
        if candidates:
            rels.append(candidates[0])
        else:
            # fabricate a link if missing for continuity
            rels.append(Relation(source=a, relationship="CONNECTS", destination=b, created_at="", updated_at="", weight=1.0, memory_id=None))
    return rels


def dijkstra_shortest_path(graph: GraphStore, start: str, goal: str, max_nodes: int = 10000) -> List[str]:
    """Weighted shortest path using edge weights."""
    import heapq
    dist: Dict[str, float] = {start: 0.0}
    prev: Dict[str, Optional[str]] = {start: None}
    pq = [(0.0, start)]
    visited = set()
    while pq and len(visited) < max_nodes:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        # expand neighbors
        for rel in graph.neighbors(node, direction="both", relationship=None, limit=1000):
            nxt = rel.destination if rel.source == node else rel.source
            if nxt in visited:
                continue
            w = graph.get_edge_weight(node, nxt)
            nd = d + w
            if nd < dist.get(nxt, float('inf')):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(pq, (nd, nxt))
    if goal not in dist:
        return []
    # reconstruct
    path: List[str] = []
    cur: Optional[str] = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path

