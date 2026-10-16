"""What every receipt records about the run that wrote it, and how a record is written.

Shared by the acquisition driver, the sample, the trainer and the generations, none of which may
import the others for it: ``environment()`` is the interpreter, the libraries, the vesper commit
and whether the tree was dirty (maintainer B's review of 66b674a: two measurements recorded a commit
that lacked the option they used); ``write_json`` writes one indented record.
"""

import json
import platform
from importlib.metadata import version
from pathlib import Path

from vesper.books import git_output

_LIBRARIES = ("pyarrow", "huggingface-hub", "tokenizers", "defusedxml", "requests")


def environment() -> dict[str, str]:
    """Interpreter, library versions, the vesper commit and whether the tree was dirty.

    ``vesper_dirty`` is ``"true"`` when tracked files differed from ``vesper_commit`` or
    untracked files were present when the record was made, so HEAD is never mistaken for the
    code that ran; ``"unknown"`` when git could not answer.
    """
    env = {"python": platform.python_version(), "vesper": version("vesper")}
    for library in _LIBRARIES:
        env[library] = version(library)
    repo = Path(__file__).resolve().parent
    try:
        env["vesper_commit"] = git_output(["rev-parse", "HEAD"], repo).strip()
        env["vesper_dirty"] = (
            "true" if git_output(["status", "--porcelain"], repo).strip() else "false"
        )
    except RuntimeError:
        env["vesper_commit"] = "unknown"
        env["vesper_dirty"] = "unknown"
    return env


def write_json(path: Path, record: dict[str, object]) -> None:
    """One receipt or listing as indented JSON, the parent directory created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
