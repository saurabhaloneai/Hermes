from pathlib import Path
from huggingface_hub import snapshot_download

def download_model_from_hf(repo_id, local_dir="./model_cache"):
    local_dir = Path(local_dir)
    local_dir.mkdir(exist_ok=True)
    model_path = snapshot_download(repo_id=repo_id, local_dir=local_dir / repo_id.replace("/", "_"), local_dir_use_symlinks=False)
    return Path(model_path)

if __name__ == "__main__":
    download_model_from_hf("Qwen/Qwen3-0.6B")