from huggingface_hub import snapshot_download

# Define the model repo and target directory
model_id = "BAAI/bge-base-en-v1.5"
target_dir = "./models/bge-base-en-v1.5"

# Download all files of the model without loading it
snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

print(f"Model files saved to {target_dir}")
