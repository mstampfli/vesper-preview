"""Shared helpers: the shrunk Vesper-S config and a fake Standard Ebooks repository."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

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
