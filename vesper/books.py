"""Books from Standard Ebooks (docs/reviews/2026-10-15-mixture-sources.md): one title, one document.

A Standard Ebooks title is a public git repository under ``github.com/standardebooks`` whose text
is XHTML under ``src/epub/text/`` in the order of the spine in ``src/epub/content.opf``. The title
universe is the organization's repository listing on a date (``list_titles``: slugs of the form
``author_title``; the organization's tool and site repositories have no underscore and are
skipped, counted). The selection is the first ``count`` slugs in blake2b order of the slug
(``select_titles``): a property of the slug alone, so a title added to the universe later never
moves another one in or out. Each selected title is fetched as one shallow, sparse clone of the two
paths the text needs (``fetch_title``), and its edition is the commit the clone resolved to.
``book_text`` reads the spine, drops the publisher's boilerplate files (title page, half title,
imprint, colophon, uncopyright, list of illustrations) and extracts the block text of the rest in
spine order; the collector (``vesper.collect.collect_books``) writes one row per title.

The XML is parsed with defusedxml with DTDs, entities and external references forbidden: the
files come from a third party's repositories. Every slug passes ``check_slug`` before it is put
in a URL, a path or a command line. Nothing here crawls the publisher's site, and the only
network calls are the listing (the GitHub API, unauthenticated, sixteen pages) and the clones.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404  (git_output: argv lists, no shell, a timeout)
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from xml.etree.ElementTree import Element, ParseError  # nosec B405  (types only; _parse defuses)

import requests
from defusedxml import DefusedXmlException
from defusedxml.ElementTree import parse as defused_parse

from vesper.publish import contained

BOOK_ORG = "standardebooks"
BOOK_SOURCE = f"github.com/{BOOK_ORG}"
BOOK_LICENSE = "public-domain-us"  # the publisher's statement; CC0 for the publisher's additions
GITHUB_API = "https://api.github.com"
GITHUB_REMOTE = f"https://github.com/{BOOK_ORG}"
PACKAGE_PATH = "src/epub/content.opf"
TEXT_PATH = "src/epub/text/"
_PAGE = 100
_SLUG = re.compile(r"[a-z0-9][a-z0-9_.-]{0,199}")
_BOILERPLATE = frozenset(
    {
        "titlepage.xhtml",
        "halftitlepage.xhtml",
        "imprint.xhtml",
        "colophon.xhtml",
        "uncopyright.xhtml",
        "loi.xhtml",
    }
)
_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "x": "http://www.w3.org/1999/xhtml",
}
_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
# leaf blocks become one line each (a line break inside one stays a line break); every other
# element is a container that is walked, or inline text that is flattened into its block
_BLOCKS = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dd", "dt", "td", "th", "figcaption"}
)
_TIMEOUT_S = 600


@dataclass(frozen=True)
class TitleListing:
    """The title universe on a date: the organization's title slugs, sorted, and the names skipped.

    A skipped name has no underscore: the organization's tools, site and manual, not a title.
    """

    titles: tuple[str, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class Edition:
    """One fetched title: its slug, the local clone and the commit the clone resolved to."""

    slug: str
    repo: Path
    commit: str


@dataclass(frozen=True)
class Spine:
    """What ``content.opf`` says: title, author, language and the text files in reading order."""

    title: str
    author: str
    language: str
    files: tuple[Path, ...]


@dataclass(frozen=True)
class Book:
    """One title's text with its attribution."""

    title: str
    author: str
    language: str
    text: str


def check_slug(slug: str) -> str:
    """PRIMITIVE: the one check a repository slug passes before it enters a URL, a path or argv.

    Lower-case letters, digits, ``_``, ``-`` and ``.``, starting with a letter or digit, at most
    200 characters: no slash, no ``..`` on its own, no whitespace, nothing a shell or a path would
    read. Returns the slug unchanged. Not a check that the repository exists.

    Raises:
        ValueError: on anything else.
    """
    if _SLUG.fullmatch(slug) is None or slug in {".", ".."} or ".." in slug:
        msg = f"not a repository slug: {slug!r}"
        raise ValueError(msg)
    return slug


def _get(url: str) -> bytes:
    """One GET of a GitHub API page with a timeout; the status is checked."""
    response = requests.get(
        url, headers={"Accept": "application/vnd.github+json"}, timeout=_TIMEOUT_S
    )
    if response.status_code != 200:  # noqa: PLR2004
        remaining = response.headers.get("X-RateLimit-Remaining", "?")
        msg = f"{url}: HTTP {response.status_code} (rate limit remaining {remaining})"
        raise RuntimeError(msg)
    return response.content


def list_titles(fetch: Callable[[str], bytes] = _get, org: str = BOOK_ORG) -> TitleListing:
    """The organization's repository names as the title universe, sorted; non-titles skipped.

    Pages of 100 through the GitHub API until a short page. A title slug is ``author_title``; a
    name without an underscore (the organization's tools, site and manual) is skipped and listed.

    Raises:
        RuntimeError: on a non-200 response (the rate limit is 60 unauthenticated calls per hour).
        TypeError: on a page whose entries are not objects with string names.
        ValueError: on a name that is not a slug.
    """
    titles: set[str] = set()
    skipped: set[str] = set()
    page = 1
    while True:
        url = f"{GITHUB_API}/orgs/{org}/repos?per_page={_PAGE}&page={page}&type=public"
        entries = cast("list[object]", json.loads(fetch(url)))
        for entry in entries:
            if not isinstance(entry, dict):
                msg = f"{url}: an entry that is not an object ({type(entry).__name__})"
                raise TypeError(msg)
            name = cast("dict[str, object]", entry).get("name")
            if not isinstance(name, str):
                msg = f"{url}: an entry without a string name"
                raise TypeError(msg)
            (titles if "_" in name else skipped).add(check_slug(name))
        if len(entries) < _PAGE:
            return TitleListing(tuple(sorted(titles)), tuple(sorted(skipped)))
        page += 1


def select_titles(titles: Sequence[str], count: int) -> list[str]:
    """The first ``count`` of ``titles`` in blake2b order of the slug (duplicates once).

    The order is a property of each slug alone (keyed hash, like ``corpus._permutation`` but of
    the name rather than an index), so the ranking of two titles never changes; which titles make
    the first ``count`` still depends on the universe (a title added with a low rank pushes the
    last one out), which is why the receipt pins the listing the selection was taken from.
    ``count`` beyond the universe returns the whole universe.

    Raises:
        ValueError: on a negative count or an invalid slug.
    """
    if count < 0:
        msg = f"count must be at least 0, got {count}"
        raise ValueError(msg)

    def rank(slug: str) -> bytes:
        return hashlib.blake2b(slug.encode(), digest_size=8).digest()

    unique = sorted({check_slug(t) for t in titles}, key=rank)
    return unique[:count]


def _git() -> str:
    git = shutil.which("git")
    if git is None:
        msg = "git is not installed; the books component needs it for shallow clones"
        raise RuntimeError(msg)
    return git


def git_output(args: Sequence[str], cwd: Path | None = None) -> str:
    """Run one git command and return its stdout; the one place this package starts a process.

    ``args`` follow the executable; an argv list, no shell, a timeout.

    Raises:
        RuntimeError: when git is missing or exits non-zero (its stderr is in the message).
    """
    argv = [_git(), *args]
    # a caller may itself run inside git (a hook sets GIT_DIR, GIT_INDEX_FILE, GIT_WORK_TREE);
    # a clone or a rev-parse here must act on its own directory, never on the caller's
    # repository, so every GIT_* variable is dropped from the child's environment
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    # the argv is a list built from constants and a checked slug; no shell is involved
    done = subprocess.run(  # noqa: S603  # nosec B603
        argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
    )
    if done.returncode != 0:
        msg = f"git {' '.join(args)}: exit {done.returncode}: {done.stderr.strip()}"
        raise RuntimeError(msg)
    return done.stdout


def fetch_title(slug: str, staging: Path, remote: str = GITHUB_REMOTE) -> Edition:
    """One shallow, sparse clone of ``<remote>/<slug>`` into ``staging/<slug>``.

    The package file and the text directory only, no history, no images. Returns the edition
    (HEAD's commit).

    A clone already at the destination is removed first: staging is this tool's own scratch.

    Raises:
        ValueError: on an invalid slug.
        RuntimeError: when git is missing or a git command fails (its stderr is in the message).
    """
    _ = check_slug(slug)
    staging.mkdir(parents=True, exist_ok=True)
    repo = staging / slug
    if repo.exists():
        shutil.rmtree(repo)
    _ = git_output(
        [
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            f"{remote}/{slug}",
            str(repo),
        ]
    )
    paths = [f"/{PACKAGE_PATH}", f"/{TEXT_PATH}"]
    _ = git_output(["sparse-checkout", "set", "--no-cone", *paths], repo)
    commit = git_output(["rev-parse", "HEAD"], repo).strip()
    return Edition(slug=slug, repo=repo, commit=commit)


def _parse(path: Path) -> Element:
    """The root of one XML file, parsed with DTDs, entities and external references forbidden.

    Raises:
        ValueError: when the file is missing, not well-formed, or uses a forbidden construct.
    """
    if not path.is_file():
        msg = f"{path}: no such file"
        raise ValueError(msg)
    try:
        tree = defused_parse(str(path), forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (ParseError, DefusedXmlException) as exc:
        msg = f"{path}: not well-formed XML or a forbidden construct ({type(exc).__name__})"
        raise ValueError(msg) from exc
    root = tree.getroot()
    if root is None:
        msg = f"{path}: no root element"
        raise ValueError(msg)
    return root


def spine(repo: Path) -> Spine:
    """``content.opf`` read: title, author, language and the spine's text files minus boilerplate.

    Raises:
        ValueError: when the package file is missing or malformed, has no title or language, a
            spine item has no manifest entry, or the package or a spine file resolves (symlinks
            followed) outside the checkout.
    """
    package = repo / PACKAGE_PATH
    inside = repo.resolve()
    contained(package, inside)
    root = _parse(package)
    title = root.findtext("opf:metadata/dc:title", default="", namespaces=_NS).strip()
    author = root.findtext("opf:metadata/dc:creator", default="", namespaces=_NS).strip()
    language = root.findtext("opf:metadata/dc:language", default="", namespaces=_NS).strip()
    if not title or not language:
        msg = f"{package}: dc:title or dc:language missing"
        raise ValueError(msg)
    hrefs: dict[str, str] = {}
    for item in root.findall("opf:manifest/opf:item", _NS):
        hrefs[item.get("id", "")] = item.get("href", "")
    files: list[Path] = []
    for ref in root.findall("opf:spine/opf:itemref", _NS):
        idref = ref.get("idref", "")
        href = hrefs.get(idref, "")
        if not href or href.startswith(("/", "..")) or ".." in Path(href).parts:
            msg = f"{package}: spine item {idref!r} has no usable manifest entry ({href!r})"
            raise ValueError(msg)
        if Path(href).name in _BOILERPLATE:
            continue
        path = package.parent / href
        contained(path, inside)
        files.append(path)
    return Spine(title=title, author=author, language=language, files=tuple(files))


class _Lines:
    """The traversal's shared inline buffer: text accumulates, structural boundaries flush."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.pieces: list[str] = []

    def text(self, value: str | None) -> None:
        if value:
            self.pieces.append(value.replace("\n", " "))

    def flush(self) -> None:
        for raw in "".join(self.pieces).split("\n"):
            line = " ".join(raw.split())
            if line:
                self.lines.append(line)
        self.pieces = []

    def line_break(self) -> None:
        self.pieces.append("\n")


def _tag(node: Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _walk(node: Element, out: _Lines) -> None:
    """One recursive traversal with a shared inline buffer; structural boundaries flush it.

    Text and tails append to the buffer in document order. Entering and leaving a block element
    (``p``, headings, ``li``, cells, captions) flushes the current line, so inline runs stay one
    line (``<li>I <em>really</em> agree<ul><li>Yes</li></ul></li>`` gives ``I really agree`` and
    ``Yes``) and blocks never run together whatever the source whitespace. ``br`` breaks the line
    inside a block; a ``noteref`` anchor is skipped with its subtree while its tail is kept; the
    policy is the same wherever the anchor sits.
    """
    tag = _tag(node)
    if tag == "br":
        out.line_break()
        return
    if tag == "a" and "noteref" in node.get(_EPUB_TYPE, ""):
        return
    is_block = tag in _BLOCKS
    if is_block:
        out.flush()
    out.text(node.text)
    for child in node:
        _walk(child, out)
        out.text(child.tail)
    if is_block:
        out.flush()


def xhtml_text(path: Path) -> str:
    """The text of one XHTML file, one line per leaf block.

    Paragraphs, headings, list items and cells are the blocks; verse lines are kept through
    ``br``, inline markup is flattened, note references are dropped, runs of whitespace collapse
    inside a line.

    Raises:
        ValueError: when the file is missing or not well-formed XHTML.
    """
    root = _parse(path)
    body = root.find("x:body", _NS)
    out = _Lines()
    _walk(body if body is not None else root, out)
    out.flush()
    return "\n".join(out.lines)


def book_text(repo: Path) -> Book:
    """The whole title as one text: its files in spine order, joined by a blank line.

    Raises:
        ValueError: from ``spine`` or ``xhtml_text``.
    """
    s = spine(repo)
    parts = [xhtml_text(path) for path in s.files]
    text = "\n\n".join(part for part in parts if part)
    return Book(title=s.title, author=s.author, language=s.language, text=text)
