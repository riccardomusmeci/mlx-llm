from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download


def download_from_hf(repo_id: str, revision: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Download entire repo from HuggingFace Hub.

    Args:
        repo_id (str): HuggingFace Hub repo id
        revision (Optional[str], optional): An optional Git revision id which can be a branch name, a tag, or a commit hash.
        filename (Optional[str], optional): An optional filename to download from the repo.

    Returns:
        str: path to downloaded weights
    """
    try:
        if filename is None:
            model_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "tokenizer.model",
                    "*.tiktoken",
                ],
                resume_download=True,
            )
        else:
            model_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename)
    except Exception as e:
        print(f"[ERROR] Downloading repo from HuggingFace Hub failed: {e}.")
        raise e

    return model_path
