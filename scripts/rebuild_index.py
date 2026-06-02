import argparse
import os
import sys

# Add project root to path to allow importing mem0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from mem0.memory import OfflineMemory
from mem0.local_config import OfflineMemoryConfig


def main():
    mem = OfflineMemory(OfflineMemoryConfig.from_file("configs/offline.yaml"))
    mem.rebuild_vector_index()
    print("Rebuilt vector index from docstore.")


if __name__ == "__main__":
    main()
