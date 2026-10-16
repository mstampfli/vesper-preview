import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vesper.collect import (
    CODE_REVISION,
    OUTPUT_COLUMNS,
    WIKI_REVISION,
    ParquetOutput,
    SpdxHeader,
    collect_code,
    collect_wikipedia,
    spdx_header,
)
from vesper.corpus import iter_raw_documents
from vesper.keys import dedup_key, file_sha256, normalize_text_v2

INDENTED = "def f():\n\treturn  'a  b'  \n"


def write_code_shard(path: Path) -> None:
    rows: dict[str, list[str | None]] = {
        "repo_name": ["o/a", "o/a", "o/b", "o/c", "o/d", "o/e", "o/f", "o/g"],
        "path": ["src/x.py", "src/y.py", "m.c", "n.rs", "p.go", "q.js", "r.py", "s.py"],
        "size": ["10", "11", "12", "13", "14", "15", "16", "17"],
        "content": [
            INDENTED,
            "print(1)\n",
            "int main() {}\n",
            "fn main() {}\n",
            "x",
            "y",
            "z",
            "w",
        ],
        "license": ["mit", "MIT", "gpl-3.0", None, "apache-2.0", "unknown", "cc0-1.0", "unlicense"],
    }
    pq.write_table(pa.Table.from_pydict(rows), str(path), row_group_size=2)


def write_wiki_shard(path: Path) -> None:
    rows: dict[str, list[str]] = {
        "id": ["1", "2", "3", "4"],
        "url": [f"https://en.wikipedia.org/wiki/{t}" for t in ("A", "B", "C", "D")],
        "title": ["A", "B", "C", "D"],
        "text": ["Alpha  beta\tgamma  \nline", "Alpha beta gamma\nline", "", "Delta"],
    }
    pq.write_table(pa.Table.from_pydict(rows), str(path), row_group_size=3)


def test_normalize_text_v2_collapses_runs_and_keeps_line_endings() -> None:
    assert normalize_text_v2("a  b\t c  \nline\n\n  x") == "a b c\nline\n\nx"
    assert normalize_text_v2("") == ""


def test_dedup_keys_are_versioned_and_code_is_exact() -> None:
    version, key = dedup_key(INDENTED, "code")
    assert version == "code-v1"
    assert key == hashlib.sha256(INDENTED.encode("utf-8")).hexdigest()
    assert dedup_key(INDENTED.replace("\t", " "), "code")[1] != key  # indentation is the program
    t_version, t_key = dedup_key("Alpha  beta\tgamma  \nline", "text")
    assert t_version == "text-v2"
    assert t_key == dedup_key("Alpha beta gamma\nline", "text")[1]
    assert t_key == hashlib.sha256(b"Alpha beta gamma\nline").hexdigest()
    with pytest.raises(ValueError, match="kind"):
        _ = dedup_key("x", "prose")


def test_collect_code_filters_by_license_and_keeps_provenance(tmp_path: Path) -> None:
    shard = tmp_path / "train-00000-of-01126.parquet"
    write_code_shard(shard)
    out = tmp_path / "code" / "00000.parquet"
    receipt = collect_code(shard, out)
    assert receipt.rows_read == 8
    assert receipt.rows_kept == 5  # mit, MIT (case-folded), apache-2.0, cc0-1.0, unlicense
    assert receipt.rejected == {"gpl-3.0": 1, "missing": 1, "unknown": 1}
    kept_texts = (INDENTED, "print(1)\n", "x", "z", "w")
    assert receipt.bytes_kept == sum(len(t.encode()) for t in kept_texts)
    assert receipt.output_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt.input_sha256 == file_sha256(shard)
    assert receipt.shard == "train-00000-of-01126.parquet"
    assert "cc0-1.0" in receipt.rule
    table = pq.ParquetFile(str(out)).read_row_group(0)
    assert table.schema.names == list(OUTPUT_COLUMNS)
    ids = table.column("id").to_pylist()
    assert ids == [
        f"code:o/a/src/x.py@{CODE_REVISION}",
        f"code:o/a/src/y.py@{CODE_REVISION}",
        f"code:o/d/p.go@{CODE_REVISION}",
        f"code:o/f/r.py@{CODE_REVISION}",
        f"code:o/g/s.py@{CODE_REVISION}",
    ]
    groups = ["code:o/a", "code:o/a", "code:o/d", "code:o/f", "code:o/g"]
    assert table.column("group_key").to_pylist() == groups
    assert table.column("license").to_pylist() == [
        "mit",
        "mit",
        "apache-2.0",
        "cc0-1.0",
        "unlicense",
    ]
    assert table.column("origin").to_pylist() == ["src/x.py", "src/y.py", "p.go", "r.py", "s.py"]
    assert table.column("header").to_pylist() == [""] * 5
    assert table.column("text").to_pylist()[0] == INDENTED  # exact, never normalized
    assert table.column("dedup_key").to_pylist()[0] == dedup_key(INDENTED, "code")[1]
    assert set(table.column("dedup_version").to_pylist()) == {"code-v1"}
    # the collected file reads like any shard of the corpus
    docs = list(iter_raw_documents([out], seed=0))
    assert {d.doc_id for d in docs} == set(ids)


def test_collect_wikipedia_keeps_attribution_and_rejects_empty(tmp_path: Path) -> None:
    shard = tmp_path / "train-00000-of-00041.parquet"
    write_wiki_shard(shard)
    out = tmp_path / "wiki" / "00000.parquet"
    receipt = collect_wikipedia(shard, out)
    assert receipt.rows_read == 4
    assert receipt.rows_kept == 3
    assert receipt.rejected == {"empty": 1}
    table = pq.ParquetFile(str(out)).read_row_group(0)
    assert table.column("id").to_pylist() == ["wiki:1", "wiki:2", "wiki:4"]
    assert table.column("group_key").to_pylist() == ["wiki:1", "wiki:2", "wiki:4"]
    assert table.column("title").to_pylist() == ["A", "B", "D"]
    assert table.column("origin").to_pylist()[0] == "https://en.wikipedia.org/wiki/A"
    assert set(table.column("license").to_pylist()) == {"cc-by-sa-3.0"}
    assert set(table.column("revision").to_pylist()) == {WIKI_REVISION}
    keys = table.column("dedup_key").to_pylist()
    assert keys[0] == keys[1]  # the two articles differ only in whitespace runs
    assert keys[0] != keys[2]
    texts = table.column("text").to_pylist()
    assert texts[0] == "Alpha  beta\tgamma  \nline"  # the training text is not rewritten


def test_collect_rejects_missing_columns_and_wrong_types(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pydict({"repo_name": ["a"], "content": ["b"]}), str(bad))
    with pytest.raises(ValueError, match="missing column"):
        _ = collect_code(bad, tmp_path / "o.parquet")
    typed = tmp_path / "typed.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {"repo_name": ["a"], "path": ["p"], "content": [1], "license": ["mit"]}
        ),
        str(typed),
    )
    with pytest.raises(TypeError, match="not str"):
        _ = collect_code(typed, tmp_path / "o2.parquet")


def test_an_empty_result_is_still_a_valid_file(tmp_path: Path) -> None:
    shard = tmp_path / "gpl.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {"repo_name": ["a"], "path": ["p"], "content": ["c"], "license": ["gpl-3.0"]}
        ),
        str(shard),
    )
    out = tmp_path / "empty.parquet"
    receipt = collect_code(shard, out)
    assert receipt.rows_kept == 0
    table = pq.ParquetFile(str(out)).read_row_group(0)
    assert table.num_rows == 0
    assert table.schema.names == list(OUTPUT_COLUMNS)


def _open_targets() -> set[str]:
    targets: set[str] = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            targets.add(str(fd.readlink()))
        except FileNotFoundError:
            continue
    return targets


def test_a_failed_run_leaves_a_prior_output_and_the_input_untouched(tmp_path: Path) -> None:
    shard = tmp_path / "ok.parquet"
    write_code_shard(shard)
    out = tmp_path / "out.parquet"
    _ = collect_code(shard, out)
    before = out.read_bytes()
    # failure before the first group: a shard without the columns
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pydict({"id": ["x"], "body": ["y"]}), str(bad))
    with pytest.raises(ValueError, match="missing column"):
        _ = collect_code(bad, out)
    assert out.read_bytes() == before
    assert str(out) not in _open_targets()
    # failure in a later group: a kept row whose content is not a string
    later = tmp_path / "later.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "repo_name": ["a", "b"],
                "path": ["p", "q"],
                "content": ["c", None],
                "license": ["mit", "mit"],
            }
        ),
        str(later),
        row_group_size=1,
    )
    with pytest.raises(TypeError, match="not str"):
        _ = collect_code(later, out)
    assert out.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "bad.parquet",
        "later.parquet",
        "ok.parquet",
        "out.parquet",
    ]  # no temporary file left behind
    # the input is never the output
    original = shard.read_bytes()
    with pytest.raises(ValueError, match="same file"):
        _ = collect_code(shard, shard)
    assert shard.read_bytes() == original


def test_hashes_are_computed_bounded_and_cover_input_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "s.parquet"
    write_code_shard(shard)
    out = tmp_path / "o.parquet"

    def whole_file(_self: Path) -> bytes:
        msg = "the collector must not materialize a whole file"
        raise AssertionError(msg)

    monkeypatch.setattr(Path, "read_bytes", whole_file)
    receipt = collect_code(shard, out)
    monkeypatch.undo()
    assert receipt.output_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt.input_sha256 == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert "apache-2.0" in receipt.rule
    assert "mit" in receipt.rule


def test_rejected_rows_need_no_string_payload(tmp_path: Path) -> None:
    shard = tmp_path / "s.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "repo_name": ["owner/a", "owner/b"],
                "path": ["ok.py", "bad.py"],
                "content": ["ok", None],
                "license": ["mit", "gpl-3.0"],
            }
        ),
        str(shard),
    )
    receipt = collect_code(shard, tmp_path / "o.parquet")
    assert (receipt.rows_kept, receipt.rejected) == (1, {"gpl-3.0": 1})


def test_an_explicit_header_outside_the_allowlist_rejects_the_file(tmp_path: Path) -> None:
    shard = tmp_path / "s.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "repo_name": ["o/r"] * 5,
                "path": ["gpl.py", "mit.py", "either.py", "both.py", "plain.py"],
                "content": [
                    "# SPDX-License-Identifier: GPL-3.0-only\nprint(1)\n",
                    "// SPDX-License-Identifier: MIT\nint x;\n",
                    "# SPDX-License-Identifier: (MIT OR Apache-2.0)\n",
                    "# SPDX-License-Identifier: MIT AND GPL-2.0-or-later\n",
                    "print(2)\n",
                ],
                "license": ["mit"] * 5,
            }
        ),
        str(shard),
    )
    out = tmp_path / "o.parquet"
    receipt = collect_code(shard, out)
    assert receipt.rows_kept == 3
    assert receipt.rejected == {"header:gpl-3.0-only": 1, "header:gpl-2.0-or-later": 1}
    table = pq.ParquetFile(str(out)).read_row_group(0)
    assert table.column("origin").to_pylist() == ["mit.py", "either.py", "plain.py"]
    assert table.column("header").to_pylist() == ["MIT", "(MIT OR Apache-2.0)", ""]


def test_normalize_text_v2_is_pinned_exactly() -> None:
    # leading and trailing Unicode whitespace of every line removed (a CRLF ending becomes LF,
    # a no-break space at either end goes), runs of spaces and tabs inside a line collapsed to
    # one space, empty lines kept; the version is v2 because v1's two implementations differed
    # on a leading no-break space (collector review, second round)
    assert normalize_text_v2(" a\r\n\tb\r\n") == "a\nb\n"
    assert normalize_text_v2("x  \t y\n\n  z") == "x y\n\nz"
    assert normalize_text_v2("\r\n") == "\n"
    nbsp = "\N{NO-BREAK SPACE}"
    assert normalize_text_v2(f"{nbsp}lead{nbsp}\nin{nbsp}side") == f"lead\nin{nbsp}side"


def test_spdx_header_names_the_licenses_an_expression_mentions() -> None:
    assert spdx_header("print(1)\n") == SpdxHeader("absent", "", ())
    assert spdx_header("/* SPDX-License-Identifier: MIT */\nint x;\n") == SpdxHeader(
        "parsed", "MIT", ("mit",)
    )
    assert spdx_header("<!-- SPDX-License-Identifier: Apache-2.0 -->") == SpdxHeader(
        "parsed", "Apache-2.0", ("apache-2.0",)
    )
    both = spdx_header("# SPDX-License-Identifier: (GPL-2.0-only OR MIT)\n")
    assert both == SpdxHeader("parsed", "(GPL-2.0-only OR MIT)", ("gpl-2.0-only", "mit"))
    # the old dual-license form names two licenses
    dual = spdx_header("# SPDX-License-Identifier: MIT/GPL-3.0-only\n")
    assert dual.identifiers == ("mit", "gpl-3.0-only")
    # only the first four kibibytes are searched
    far = spdx_header("x" * 5000 + "\n# SPDX-License-Identifier: GPL-3.0-only\n")
    assert far.status == "absent"


def test_spdx_header_reads_strictly_and_refuses_what_it_cannot_name() -> None:
    # seen in the first four github-code shards on 2026-10-15: a sentence's period is allowed,
    # the deprecated plus form keeps its plus; present but unreadable evidence (markup after the
    # identifier, a bare "***", an exception clause) is "unsupported", never approval (collector
    # review, fourth round)
    assert spdx_header("# SPDX-License-Identifier: Apache-2.0.\n") == SpdxHeader(
        "parsed", "Apache-2.0.", ("apache-2.0",)
    )
    assert spdx_header("// SPDX-License-Identifier: GPL-2.0+\n").identifiers == ("gpl-2.0+",)
    quoted = "<div>SPDX-License-Identifier: BSD-3-Clause</span></div>\n"
    assert spdx_header(quoted) == SpdxHeader("unsupported", "BSD-3-Clause</span></div>", ())
    assert spdx_header("# SPDX-License-Identifier: ***\n").status == "unsupported"
    assert spdx_header("# SPDX-License-Identifier: MIT WITH Unknown-Exception\n").status == (
        "unsupported"
    )
    assert spdx_header("# SPDX-License-Identifier:\n").status == "unsupported"


def test_two_writers_on_one_destination_own_their_temporaries(tmp_path: Path) -> None:
    # Maintainer B's fixture (collector review, third round): two interleaved writers shared one
    # ".<name>.partial"; the second mutated the first's published file
    out = tmp_path / "shared.parquet"
    row = dict.fromkeys(OUTPUT_COLUMNS, "")
    first = ParquetOutput(out)
    second = ParquetOutput(out)
    assert first.temp != second.temp
    first.add({**row, "id": "one", "text": "first"})
    second.add({**row, "id": "two", "text": "second"})
    digest_first = first.close()
    assert digest_first == file_sha256(out)
    assert pq.ParquetFile(str(out)).read_row_group(0).column("id").to_pylist() == ["one"]
    digest_second = second.close()
    assert digest_second == file_sha256(out)
    assert pq.ParquetFile(str(out)).read_row_group(0).column("id").to_pylist() == ["two"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.parquet"]  # no temporaries


def test_a_failed_publication_leaves_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o.parquet"
    writer = ParquetOutput(out)
    writer.add(dict.fromkeys(OUTPUT_COLUMNS, "") | {"id": "x", "text": "y"})

    def refuse(_self: Path, _target: Path) -> Path:
        msg = "no rename today"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="no rename"):
        _ = writer.close()
    monkeypatch.undo()
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
    writer.abort()  # idempotent after a failed close


def test_a_writers_hash_is_its_own_content_even_when_another_publishes_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture (collector review, sixth round): a second publication between the first
    # writer's rename and its hash made the first writer report the second's hash
    out = tmp_path / "shared.parquet"
    row = dict.fromkeys(OUTPUT_COLUMNS, "")
    first = ParquetOutput(out)
    second = ParquetOutput(out)
    first.add({**row, "id": "one", "text": "first"})
    second.add({**row, "id": "two", "text": "second"})
    real_replace = Path.replace
    published_by_second: list[str] = []

    def interleaved(self: Path, target: Path) -> Path:
        result = real_replace(self, target)
        if self == first.temp and not published_by_second:
            # the other writer publishes right after the first rename, before any hash
            published_by_second.append(second.close())
        return result

    monkeypatch.setattr(Path, "replace", interleaved)
    digest_first = first.close()
    monkeypatch.undo()
    assert published_by_second  # the race happened
    assert pq.ParquetFile(str(out)).read_row_group(0).column("id").to_pylist() == ["two"]
    assert digest_first != file_sha256(out)  # the first writer reports its own content
    assert published_by_second[0] == file_sha256(out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.parquet"]


def test_a_failed_hash_leaves_no_temporary_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture (collector review, seventh round): cleanup keyed on the open handle
    # left the temporary behind when hashing it failed after the writer had closed
    out = tmp_path / "o.parquet"
    writer = ParquetOutput(out)
    writer.add(dict.fromkeys(OUTPUT_COLUMNS, "") | {"id": "x", "text": "y"})

    def refuse(_path: Path) -> str:
        msg = "disk gone"
        raise OSError(msg)

    monkeypatch.setattr("vesper.collect.file_sha256", refuse)
    with pytest.raises(OSError, match="disk gone"):
        _ = writer.close()
    monkeypatch.undo()
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
