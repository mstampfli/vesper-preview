import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from vesper.books import (
    BOOK_LICENSE,
    BOOK_SOURCE,
    GITHUB_REMOTE,
    Edition,
    book_text,
    check_slug,
    fetch_title,
    git_output,
    list_titles,
    select_titles,
    spine,
    xhtml_text,
)
from vesper.collect import OUTPUT_COLUMNS, collect_books
from vesper.corpus import iter_raw_documents
from vesper.keys import dedup_key

Builder = Callable[[Path, str, dict[str, str]], Path]


def blake_rank(slug: str) -> bytes:
    return hashlib.blake2b(slug.encode(), digest_size=8).digest()


CHAPTER = """<section epub:type="chapter">
<hgroup>
<h2 epub:type="z3998:ordinal z3998:roman">I</h2>
<p epub:type="title">In Which Something   Happens</p>
</hgroup>
<p>One day <em>quite</em> early,<a href="endnotes.xhtml#note-1" epub:type="noteref">1</a>
it snowed.</p>
<blockquote epub:type="z3998:song">
<p>
<span>The more it snows</span>
<br/>
<span class="i2">(Tiddely pom),</span>
</p>
<p>The more it goes</p>
</blockquote>
<table><tbody><tr><td>left   cell</td><td>right cell</td></tr></tbody></table>
<hr/>
<p>
   After the   break.
</p>
</section>"""

EXPECTED = (
    "I\n"
    "In Which Something Happens\n"
    "One day quite early, it snowed.\n"
    "The more it snows\n"
    "(Tiddely pom),\n"
    "The more it goes\n"
    "left cell\n"
    "right cell\n"
    "After the break."
)


def test_check_slug_accepts_titles_and_rejects_anything_a_path_or_shell_reads() -> None:
    for good in ("a-a-milne_the-house-at-pooh-corner", "x_1.2", "0_a"):
        assert check_slug(good) == good
    for bad in ("", "A_b", "a/b", "a b", "../x", "a..b", "-a", ".", "..", "a" * 201, "a_b\n"):
        with pytest.raises(ValueError, match="slug"):
            _ = check_slug(bad)


def test_list_titles_pages_until_short_and_skips_names_without_underscore() -> None:
    pages = {
        1: [{"name": f"a_{i:03d}"} for i in range(98)] + [{"name": "tools"}, {"name": "web"}],
        2: [{"name": "z_last"}, {"name": "a_000"}],
    }
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        page = int(url.rsplit("page=", 1)[1].split("&", 1)[0])
        return json.dumps(pages[page]).encode()

    listing = list_titles(fetch)
    assert len(seen) == 2
    assert "page=1&" in seen[0]
    assert "page=2&" in seen[1]
    assert listing.titles == tuple(sorted({f"a_{i:03d}" for i in range(98)} | {"z_last"}))
    assert listing.skipped == ("tools", "web")

    def bad(url: str) -> bytes:
        del url
        return b'[{"nom": 1}]'

    with pytest.raises(TypeError, match="name"):
        _ = list_titles(bad)


def test_select_titles_is_a_property_of_each_slug() -> None:
    universe = [f"a_{i}" for i in range(50)]
    chosen = select_titles(universe, 10)
    assert len(chosen) == 10
    assert select_titles(list(reversed(universe)), 10) == chosen
    # the ranking of two titles never changes, so the chosen ten are the first ten of the
    # eleven chosen from the universe plus one; which titles make a fixed count still depends
    # on the universe: a new title that ranks first pushes the tenth out (the receipt pins the
    # listing for exactly that reason)
    assert select_titles([*universe, "z_new"], 11)[:10] == chosen or "z_new" in select_titles(
        [*universe, "z_new"], 11
    )
    newcomer = min((f"n_{i}" for i in range(200)), key=blake_rank)
    with_newcomer = select_titles([*universe, newcomer], 10)
    if with_newcomer[0] == newcomer:
        assert set(with_newcomer) != set(chosen)
        assert set(with_newcomer) - {newcomer} == set(chosen[:9])
    assert select_titles(["a_1", "a_1"], 5) == ["a_1"]
    assert len(select_titles(universe, 500)) == 50
    with pytest.raises(ValueError, match="count"):
        _ = select_titles(universe, -1)

    def rank(slug: str) -> bytes:
        return hashlib.blake2b(slug.encode(), digest_size=8).digest()

    three = ["b_x", "a_y", "c_z"]
    assert select_titles(three, 3) == sorted(three, key=rank)


def test_spine_keeps_the_reading_order_and_drops_the_boilerplate(
    tmp_path: Path, book_builder: Builder
) -> None:
    repo = book_builder(
        tmp_path, "en-GB", {"chapter-2.xhtml": "<p>b</p>", "chapter-1.xhtml": "<p>a</p>"}
    )
    s = spine(repo)
    assert (s.title, s.author, s.language) == ("The Book", "A. Writer", "en-GB")
    assert [p.name for p in s.files] == ["chapter-2.xhtml", "chapter-1.xhtml"]


def test_spine_rejects_a_missing_package_a_dangling_item_and_a_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no such file"):
        _ = spine(tmp_path)
    epub = tmp_path / "src" / "epub"
    epub.mkdir(parents=True)
    opf = epub / "content.opf"
    head = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
        "<dc:language>en</dc:language></metadata>"
    )
    _ = opf.write_text(head + '<manifest/><spine><itemref idref="x"/></spine></package>')
    with pytest.raises(ValueError, match="manifest"):
        _ = spine(tmp_path)
    traversal = '<manifest><item id="x" href="../../etc/passwd"/></manifest>'
    _ = opf.write_text(head + traversal + '<spine><itemref idref="x"/></spine></package>')
    with pytest.raises(ValueError, match="manifest"):
        _ = spine(tmp_path)
    package = '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
    metadata = '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:language>en</dc:language>'
    untitled = package + metadata + "</metadata><manifest/><spine/></package>"
    _ = opf.write_text(untitled)
    with pytest.raises(ValueError, match="dc:title"):
        _ = spine(tmp_path)


def test_xhtml_text_blocks_verse_notes_and_whitespace(
    tmp_path: Path, book_builder: Builder
) -> None:
    repo = book_builder(tmp_path, "en", {"chapter-1.xhtml": CHAPTER})
    assert xhtml_text(repo / "src" / "epub" / "text" / "chapter-1.xhtml") == EXPECTED


def test_xhtml_text_refuses_a_dtd_an_entity_and_a_malformed_file(tmp_path: Path) -> None:
    bomb = tmp_path / "bomb.xhtml"
    entities = (
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaaaaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
    )
    _ = bomb.write_text(entities + "<html><body><p>&b;</p></body></html>")
    with pytest.raises(ValueError, match="forbidden"):
        _ = xhtml_text(bomb)
    doctype = tmp_path / "doctype.xhtml"
    _ = doctype.write_text("<!DOCTYPE html><html><body><p>x</p></body></html>")
    with pytest.raises(ValueError, match="forbidden"):
        _ = xhtml_text(doctype)
    broken = tmp_path / "broken.xhtml"
    _ = broken.write_text("<html><body><p>unclosed</body></html>")
    with pytest.raises(ValueError, match="well-formed"):
        _ = xhtml_text(broken)
    with pytest.raises(ValueError, match="no such file"):
        _ = xhtml_text(tmp_path / "missing.xhtml")


def test_book_text_joins_files_with_a_blank_line_and_skips_empty_ones(
    tmp_path: Path, book_builder: Builder
) -> None:
    repo = book_builder(
        tmp_path,
        "en-US",
        {"chapter-1.xhtml": "<p>A</p>", "blank.xhtml": "<p>  </p>", "chapter-2.xhtml": "<p>B</p>"},
    )
    book = book_text(repo)
    assert book.text == "A\n\nB"
    assert (book.title, book.author, book.language) == ("The Book", "A. Writer", "en-US")


def test_collect_books_keeps_english_titles_with_provenance_and_counts_the_rest(
    tmp_path: Path, book_builder: Builder
) -> None:
    english = book_builder(
        tmp_path / "e",
        "en-GB",
        {"chapter-1.xhtml": "<p>Alpha  beta</p>", "chapter-2.xhtml": "<p>Gamma</p>"},
    )
    french = book_builder(tmp_path / "f", "fr", {"chapter-1.xhtml": "<p>Bonjour</p>"})
    empty = book_builder(tmp_path / "m", "en", {"chapter-1.xhtml": "<p> </p>"})
    broken = tmp_path / "b"
    broken.mkdir()
    editions = [
        Edition("e_book", english, "e1"),
        Edition("f_livre", french, "f1"),
        Edition("m_blank", empty, "m1"),
        Edition("b_broken", broken, "b1"),
    ]
    out = tmp_path / "books" / "titles.parquet"
    result = collect_books(iter(editions), out, "listing-x")
    receipt = result.receipt
    assert (receipt.rows_read, receipt.rows_kept) == (4, 1)
    assert receipt.rejected == {"language": 1, "empty": 1, "malformed": 1}
    text = book_text(english).text
    assert text == "Alpha beta\n\nGamma"  # the extraction collapses runs; the collector keeps it
    assert receipt.bytes_kept == len(text.encode("utf-8"))
    assert (receipt.source, receipt.revision, receipt.shard) == (
        BOOK_SOURCE,
        "listing-x",
        "titles:4",
    )
    assert receipt.output_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = "\n".join(sorted(f"{e.slug}@{e.commit}" for e in editions)).encode()
    assert receipt.input_sha256 == hashlib.sha256(manifest).hexdigest()
    assert "en" in receipt.rule
    assert [t.status for t in result.titles] == ["kept", "language", "empty", "malformed"]
    assert result.titles[0].note == "A. Writer"
    assert result.titles[1].note == "fr"
    assert "no such file" in result.titles[3].note
    table = pq.ParquetFile(str(out)).read_row_group(0)
    assert table.schema.names == list(OUTPUT_COLUMNS)
    assert table.column("id").to_pylist() == ["book:e_book@e1"]
    assert table.column("group_key").to_pylist() == ["book:e_book"]
    assert table.column("revision").to_pylist() == ["e1"]
    assert table.column("license").to_pylist() == [BOOK_LICENSE]
    assert table.column("origin").to_pylist() == [f"{GITHUB_REMOTE}/e_book"]
    assert table.column("title").to_pylist() == ["The Book"]
    assert table.column("text").to_pylist() == [text]  # what the extraction gave, unchanged
    assert table.column("dedup_key").to_pylist() == [dedup_key(text, "text")[1]]
    assert {d.doc_id for d in iter_raw_documents([out], seed=0)} == {"book:e_book@e1"}


def _git(args: list[str], cwd: Path) -> str:
    # the suite runs inside the pre-commit hook, where git exports GIT_DIR and GIT_INDEX_FILE;
    # a test's git must act on its own temporary repository, never on the project's
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    done = subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],  # noqa: S607
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_git_output_ignores_the_callers_repository_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a pre-commit hook exports GIT_DIR and GIT_INDEX_FILE; on 2026-10-15 a test's git init,
    # add and commit inside the hook acted on the project repository through them
    own = tmp_path / "own"
    own.mkdir()
    _ = _git(["init", "-q"], own)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _ = _git(["init", "-q"], elsewhere)
    monkeypatch.setenv("GIT_DIR", str(elsewhere / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(elsewhere / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(elsewhere))
    top = Path(git_output(["rev-parse", "--show-toplevel"], own).strip())
    assert top.resolve() == own.resolve()


def test_fetch_title_clones_the_text_only_and_reports_the_commit(
    tmp_path: Path, book_builder: Builder
) -> None:
    remote = tmp_path / "remote"
    source = book_builder(remote / "x_book", "en", {"chapter-1.xhtml": "<p>Text</p>"})
    (source / "images").mkdir()
    _ = (source / "images" / "cover.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    _ = _git(["init", "-q"], source)
    _ = _git(["config", "uploadpack.allowFilter", "true"], source)
    _ = _git(["add", "-A"], source)
    _ = _git(["commit", "-q", "-m", "edition"], source)
    head = _git(["rev-parse", "HEAD"], source)
    staging = tmp_path / "staging"
    edition = fetch_title("x_book", staging, remote=f"file://{remote}")
    assert edition == Edition("x_book", staging / "x_book", head)
    assert (edition.repo / "src" / "epub" / "content.opf").is_file()
    assert (edition.repo / "src" / "epub" / "text" / "chapter-1.xhtml").is_file()
    assert not (edition.repo / "images").exists()
    assert book_text(edition.repo).text == "Text"
    # a second fetch replaces the clone in staging
    assert fetch_title("x_book", staging, remote=f"file://{remote}") == edition
    with pytest.raises(ValueError, match="slug"):
        _ = fetch_title("../x", staging, remote=f"file://{remote}")
    with pytest.raises(RuntimeError, match="clone"):
        _ = fetch_title("y_missing", staging, remote=f"file://{remote}")
    with pytest.raises(RuntimeError, match="exit"):
        _ = git_output(["rev-parse", "--verify", "nothing-here"], edition.repo)


def test_spine_refuses_a_file_that_resolves_outside_the_checkout(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's fixture (collector review, second round): a manifest href under text/ that is a
    # symlink to a file outside the clone; the lexical checks on the href do not see it
    repo = book_builder(tmp_path / "repo", "en", {"chapter-1.xhtml": "<p>Inside</p>"})
    sentinel = tmp_path / "outside.xhtml"
    _ = sentinel.write_text(XHTML_LIKE)
    link = repo / "src" / "epub" / "text" / "chapter-1.xhtml"
    link.unlink()
    link.symlink_to(sentinel)
    with pytest.raises(ValueError, match="resolves outside"):
        _ = spine(repo)
    with pytest.raises(ValueError, match="resolves outside"):
        _ = book_text(repo)
    # the package file itself as a symlink is refused the same way
    other = book_builder(tmp_path / "other", "en", {"chapter-1.xhtml": "<p>Other</p>"})
    package = other / "src" / "epub" / "content.opf"
    moved = tmp_path / "moved.opf"
    _ = package.rename(moved)
    package.symlink_to(moved)
    with pytest.raises(ValueError, match="resolves outside"):
        _ = spine(other)


XHTML_LIKE = (
    '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
    "<body><p>SENTINEL</p></body></html>"
)


def test_nested_blocks_become_their_own_lines(tmp_path: Path, book_builder: Builder) -> None:
    # Maintainer B's fixture (collector review, third round): minified valid XHTML with blocks
    # inside blocks ran paragraphs together as "OneTwo" and "ParentChild"
    body = (
        "<ul><li><p>One</p><p>Two</p></li></ul>"
        "<div><p>Parent</p><p>Child</p></div>"
        "<li>Intro<p>Para</p>tail</li>"
        "<p>Plain<em>inline</em>text</p>"
    )
    repo = book_builder(tmp_path, "en", {"chapter-1.xhtml": body})
    lines = xhtml_text(repo / "src" / "epub" / "text" / "chapter-1.xhtml").split("\n")
    assert lines == ["One", "Two", "Parent", "Child", "Intro", "Para", "tail", "Plaininlinetext"]


def test_blocks_below_an_intervening_element_are_still_boundaries(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's fixture (collector review, fifth round): the nested list's inner block sits
    # under a ul, not directly under the li; and an inline element directly inside a container is
    # text
    body = "<ul><li>Parent<ul><li>Child</li></ul></li></ul><div><em>Parent</em><p>Child</p></div>"
    repo = book_builder(tmp_path, "en", {"chapter-1.xhtml": body})
    lines = xhtml_text(repo / "src" / "epub" / "text" / "chapter-1.xhtml").split("\n")
    assert lines == ["Parent", "Child", "Parent", "Child"]


def test_inline_runs_stay_one_line_and_note_references_are_dropped_everywhere(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's fixtures (collector review, seventh round): inline text around a nested list is
    # one line, and a note reference is skipped wherever it sits, with its tail kept
    body = (
        "<ul><li>I <em>really</em> agree<ul><li>Yes</li></ul></li></ul>"
        '<ul><li>Text<a epub:type="noteref" href="#n1">1</a> goes on'
        "<ul><li>Child</li></ul></li></ul>"
        '<p>Plain<a epub:type="noteref" href="#n2">2</a> text</p>'
        "<p>Line<br/>break</p>"
    )
    repo = book_builder(tmp_path, "en", {"chapter-1.xhtml": body})
    lines = xhtml_text(repo / "src" / "epub" / "text" / "chapter-1.xhtml").split("\n")
    assert lines == [
        "I really agree",
        "Yes",
        "Text goes on",
        "Child",
        "Plain text",
        "Line",
        "break",
    ]
    spaced = book_builder(
        tmp_path / "spaced",
        "en",
        {
            "chapter-1.xhtml": (
                "<ul>\n  <li>\n    I <em>really</em> agree\n    <ul>\n      <li>Yes</li>\n"
                "    </ul>\n  </li>\n</ul>"
            )
        },
    )
    assert xhtml_text(spaced / "src" / "epub" / "text" / "chapter-1.xhtml").split("\n") == [
        "I really agree",
        "Yes",
    ]
