import argparse
import os
import sys

# Add project root to path to allow importing mem0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mem0.graph_store import GraphStore
from mem0.local_config import LocalGraphConfig, OfflineMemoryConfig
from mem0.visualize import export_pyvis


def main():
    cfg = OfflineMemoryConfig.from_file("configs/offline.yaml")
    g = GraphStore(LocalGraphConfig(db_path=cfg.graph.db_path))
    all_nodes = g.query_nodes()
    entities = [n.name for n in all_nodes if getattr(n, 'name', None)]
    if not entities:
        print('[viz] no entities found in graph')
        return
    path = export_pyvis(g, nodes=entities, edges=None)
    print(f"[viz_export] exported entire graph ({len(entities)} entities) to {path}")


if __name__ == '__main__':
    main()
