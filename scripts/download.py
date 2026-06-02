from __future__ import annotations

import argparse
import os
from typing import List

from huggingface_hub import snapshot_download


MODELS = {
    "reranker": {
        "repo_id": "Qwen/Qwen3-Reranker-0.6B",
        "target": "Qwen3-Reranker-0.6B",
    },
    "embed": {
        "repo_id": "jinaai/jina-embedding-l-en-v1",
        "target": "jina-embedding-l-en-v1",
    },
    "gguf": {
        "repo_id": "roleplaiapp/AceInstruct-7B-Q4_0-GGUF",
        "target": "AceInstruct-7B-Q4_0-GGUF",
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_repo(repo_id: str, dest_dir: str) -> str:
    ensure_dir(dest_dir)
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download required models into ./models")
    parser.add_argument(
        "--which",
        default="all",
        choices=["all", "reranker", "embed", "gguf"],
        help="Which model(s) to download",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.getcwd(), "models"),
        help="Target directory for models",
    )
    args = parser.parse_args()

    plan: List[str]
    if args.which == "all":
        plan = ["reranker", "embed", "gguf"]
    else:
        plan = [args.which]

    for key in plan:
        spec = MODELS[key]
        dest = os.path.join(args.models_dir, spec["target"])
        print(f"Downloading {spec['repo_id']} -> {dest}")
        out = download_repo(spec["repo_id"], dest)
        print(f"  done: {out}")


if __name__ == "__main__":
    main()
