# Precise local stub for the one call this project makes into `huggingface_hub`
# (vesper/collect.py). The library's own hints carry Unknown in places the strict gate forbids.

def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    repo_type: str | None = ...,
    revision: str | None = ...,
    cache_dir: str | None = ...,
) -> str: ...
