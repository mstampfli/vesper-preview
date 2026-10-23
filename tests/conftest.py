"""Shared helpers: the shrunk Vesper-S config and a fake Standard Ebooks repository."""

import contextlib
import io
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

import pytest
import yaml

from vesper.config import VesperConfig, parse

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def small_raw(vocab: int = 64) -> dict[str, dict[str, object]]:
    """The raw mapping of ``small_config``, for tests that write a config file."""
    with (CONFIGS / "vesper-s.yaml").open(encoding="utf-8") as fh:
        raw = cast("dict[str, dict[str, object]]", yaml.safe_load(fh))
    raw["model"].update(
        {
            "d": 32,
            "heads": 2,
            "head_dim": 16,
            "columns": 2,
            "t_max": 2,
            "vocab": vocab,
            "window": 8,
            "kv_groups": 2,
            "ffn_mult": 2,
        }
    )
    raw["recurrence"]["block"] = 4
    raw["training"].update({"chunk_tokens": 16, "batch_chunks": 2, "t_max_anneal_from": 1})
    return raw


def small_config(vocab: int = 64) -> VesperConfig:
    """Vesper-S's file with the shapes shrunk to test size; every relation still validated."""
    return parse(small_raw(vocab), "test")


XHTML = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>t</title></head>
<body epub:type="bodymatter">
{body}
</body>
</html>
"""

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="uid">https://standardebooks.org/ebooks/x</dc:identifier>
<dc:title id="title">The Book</dc:title>
<dc:creator id="author">A. Writer</dc:creator>
<dc:language>{language}</dc:language>
</metadata>
<manifest>
{items}
</manifest>
<spine>
{refs}
</spine>
</package>
"""

BOILERPLATE = ("titlepage.xhtml", "imprint.xhtml", "colophon.xhtml", "uncopyright.xhtml")


def write_book(root: Path, language: str, chapters: dict[str, str]) -> Path:
    """A repository at ``root`` with the publisher's boilerplate files around ``chapters``.

    ``chapters`` maps file name to body XHTML, in spine order; the boilerplate files carry a
    paragraph each so a collector that forgot to drop them would show it.
    """
    epub = root / "src" / "epub"
    text = epub / "text"
    text.mkdir(parents=True)
    names: list[str] = [BOILERPLATE[0], BOILERPLATE[1], *chapters, BOILERPLATE[2], BOILERPLATE[3]]
    for name in BOILERPLATE:
        boilerplate = XHTML.format(body=f"<p>Boilerplate {name}</p>")
        _ = (text / name).write_text(boilerplate, encoding="utf-8")
    for name, body in chapters.items():
        _ = (text / name).write_text(XHTML.format(body=body), encoding="utf-8")
    items = "\n".join(
        f'<item href="text/{n}" id="{n}" media-type="application/xhtml+xml"/>' for n in names
    )
    refs = "\n".join(f'<itemref idref="{n}"/>' for n in names)
    _ = (epub / "content.opf").write_text(
        OPF.format(language=language, items=items, refs=refs), encoding="utf-8"
    )
    return root


@pytest.fixture(name="book_builder")
def _book_builder() -> Callable[[Path, str, dict[str, str]], Path]:
    return write_book


class _Open(Protocol):
    def __call__(self, file: object, mode: str = ..., /, *args: object, **kwargs: object) -> object:
        """``io.open``'s call shape, as the patch below passes it through."""
        ...


def _write(path: Path, data: bytes) -> None:
    """``data`` into ``path`` at the OS level: never through the patched ``open``."""
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
    try:
        _ = os.write(fd, data)
    finally:
        os.close(fd)


@contextlib.contextmanager
def replaced_after_first_read(path: Path, replacement: bytes) -> Generator[list[str]]:
    """``path`` replaced on disk by ``replacement`` right after its FIRST open for reading.

    The first open (through ``open`` or ``Path``'s readers) gets the original bytes; the file
    on disk then holds ``replacement``, so any later read, by Python or by a library reading
    the file itself, sees the replacement: a file swapped between a hash and a reread (maintainer
    B's reviews of f55bfb3 and 72c9d31). Yields the modes of the opens seen; the original bytes
    are restored on exit.
    """
    target = path.resolve()
    original = path.read_bytes()
    real = cast("_Open", io.open)  # the original, called for every other file
    opens: list[str] = []

    def fake(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        if isinstance(file, (str, os.PathLike)) and Path(cast("str", file)).resolve() == target:
            opens.append(mode)
            if len(opens) == 1:
                _write(target, replacement)
                return io.BytesIO(original) if "b" in mode else io.StringIO(original.decode())
        return real(file, mode, *args, **kwargs)

    try:
        with patch("io.open", fake), patch("builtins.open", fake):
            yield opens
    finally:
        _ = target.write_bytes(original)
