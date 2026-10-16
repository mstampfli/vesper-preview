# Decision: the sources of the pretraining mix and the contracts of the collector

Date: 2026-10-15. Proposal: maintainer B (channel, source proposal with checksums, 00:16Z). Decisions:
Maintainer A, maintainer; the project owner may override. Status: contracts for the collector that does not exist yet;
nothing below is a claim that a corpus has been acquired.

## Why

The spec (Data) trains on a public web-text corpus with a quality filter plus a code slice at
15% of tokens plus Wikipedia and books at 5%, and trains the tokenizer on a 10 GB sample of the
mix. The box holds the web component (FineWeb-Edu `sample/10BT`, 14 parquet files, 27 GB). The
tokenizer trained on 2026-10-15 is therefore a web-only pilot (receipt beside its artifact);
the frozen tokenizer needs the other components.

## Sources

| Component | Source | Pin | License handling |
|---|---|---|---|
| Code, 15% of tokens | `codeparrot/github-code` on Hugging Face, parquet shards (physical columns `repo_name, path, size, content, license`; the card's `code` field is `content` on disk, no language column) | revision `b5661e6b17396364b2bcf8e68977b0d28e1ebd19` | Allowlist on the repository-level label: `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `isc`, `cc0-1.0`, `unlicense` (the last two added 2026-10-15 with maintainer B's support: public-domain dedications, [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en), [Unlicense](https://unlicense.org/)); anything else, unknown or conflicting is rejected. Conflicting means an explicit `SPDX-License-Identifier` header in the file naming a license outside the allowlist. The label is evidence, not a file audit; labels, headers and paths are kept in the provenance, and every receipt records the allowlist that ran |
| Encyclopedia, part of 5% | `wikimedia/wikipedia`, `20231101.en` (41 shards; `id, url, title, text`) | revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC BY-SA 3.0 / GFDL: attribution and source URLs kept in the provenance; not treated as public domain |
| Books, part of 5% | Standard Ebooks, a bounded title-level selection from its public source repositories, text from XHTML in spine order; Project Gutenberg through its published robot and mirror routes as the fallback | per-title edition revision in the manifest | US public domain is the publisher's statement; titles are selected individually; no site crawl |

First proposal for the inside of the 5%: four parts encyclopedia to one part books.

## Contracts

- **Quotas.** The spec's 15% and 5% are token proportions measured under the frozen
  tokenizer. The tokenizer's own sample is drawn by bytes, a separate rule stated here: 80% web,
  15% code, 5% encyclopedia and books, by UTF-8 bytes of the 10 GB.
- **Identifiers.** Namespaced across sources: `web:<urn>`, `code:<repo_name>/<path>@<revision>`,
  `wiki:<id>`, `book:<slug>@<edition>`. Source identity is kept separate from the split key.
- **Grouping before the split.** The held-out split (`is_held_out`) is taken on the group key,
  not the document id: the repository for code, the title for books, the document id for web
  and encyclopedia. Related code of one repository and the chapters of one book land on the same
  side.
- **Deduplication.** A versioned, per-source dedup key decides duplicates; duplicates share a
  partition and are counted once; computed before the split, never after. Version 1 of the keys:
  code, the exact text with its line endings (indentation and spaces inside literals are the
  program, nothing is normalized); web, encyclopedia and books, `normalize_text_v2`: the text
  split on the line feed, in every line runs of spaces and tabs collapsed to one space and leading
  and trailing Unicode whitespace removed (so CRLF endings become LF and a no-break space at
  either end goes), empty lines kept, joined with the line feed. Version 1 was retired on
  2026-10-15: maintainer B's review found its wording looser than the code and then that its two
  implementations differed on a leading no-break space; one version carries one rule, so the
  clean rule is v2 and every text key is recomputed. Keys of different versions never match:
  text sources deduplicate against each other, code among code; byte-identical content in a code
  file and a text document is two documents, and that is the whole cross-source claim. The key is
  used for grouping only; the training text is never rewritten by it. Finalization is decided
  once, against the pinned inputs and independently of any shuffle (`vesper.corpus.finalize`):
  for every key occurring more than once across all components the document with the smallest
  id survives and the others are dropped (a duplicate across groups on different sides: the
  survivor's group decides, the other copies are in no split); the dropped set and the inputs'
  hashes are written with a receipt, and every reader goes through `select_documents` with a
  `Split` of that finalization, so the same document is on the same side under every seed. A
  collected parquet file is staging until a finalization covers it.
- **Acquisition.** One compressed shard at a time; selected rows written to compact local
  parquet with provenance columns; staging released before the next shard; free disk read and
  checkpoint space reserved before starting; nothing downloaded in full.
- **Receipts.** Every tokenizer run and every collected component carries a receipt: library and
  interpreter versions, repository commit, ordered input manifest with sizes and hashes, seed,
  split rule and fraction, exact UTF-8 bytes and document counts, duration, outputs with hashes.

## Registry rows (added 2026-10-15; the collector exists)

`collect_code`, `collect_wikipedia`, `collect_books` and `spdx_header` (module `collect`),
`normalize_text_v1`, `dedup_key` and `file_sha256` (module `keys`), `select_documents`,
`string_value` and `string_column` (module `corpus`), the Standard Ebooks functions (module
`books`) and the driver (module `acquire`), each with the test that rejects a wrong one: a
rejected license label, a conflicting header, a duplicate split across sides, a group split
across sides, a missing provenance column, a failed run that touched a prior output.
