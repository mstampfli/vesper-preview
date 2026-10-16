# Independent corpus review

Base: f9ce16e6edd32028d477591a4706e17bdb6da15f.
Reviewer: maintainer B. Time: 2026-10-15T00:03:27+00:00.
Status: changes requested. No source edited; full gate green, 74 tests.

## Bounded file resources

iter_raw_documents constructs every ParquetFile before yielding. An isolated
subprocess with RLIMIT_NOFILE=64 over 80 tiny valid files raises OSError 24
(EMFILE) at file 61. The fixture's documents are otherwise readable. This
establishes an input-size-dependent resource failure, not a failure of maintainer A's
current 14-file run.

Requested: discover metadata one file at a time, close each reader, then read
shuffled row groups through a bounded number of open readers. Close on normal
completion, early generator close and exceptions. A small resource primitive
and regression with more files than the available descriptor limit should
cover this boundary. Do not raise the process limit to hide linear resource
retention. Document the memory bound in terms of largest row group plus
metadata/permutation storage; arbitrary input files do not imply five MB.

## Shuffle and test claims

The algorithm permutes row-group blocks, then the rows within each block.
Every group's rows remain contiguous. The test comment claiming otherwise is
false; its disjunction passes when row order differs while still contiguous.
Name and test the block-shuffle contract and its limited mixing explicitly.
It need not be replaced with a uniform whole-corpus shuffle to fix this claim.

The test called across-process repeats in one process. Independent subprocesses
with PYTHONHASHSEED 0, 1 and random produce the same order at a fixed seed and
ordered path manifest. This validates the current implementation on those
processes, while exposing a gap in the checked-in regression. Run actual child
processes there, or narrow the name and claim. Manifest order is part of the
input: the file index enters the ranking key, so reproducibility needs it.

A noncryptographic PRNG is appropriate for ordinary dataset shuffling. A scanner
warning alone does not establish a security flaw. Keeping hash ranking is an
acceptable explicit reproducibility choice, with its sorting cost understood.

## Tokenizer run provenance and scope

Maintainer A reports a run over 10 GiB of training-split FineWeb-Edu text, begun before
this review. Maintainer B did not access its private files or logs. The spec requests
a 10 GB sample of a mixture including 15% code and 5% Wikipedia/books, so the
reported web-only run is a pilot unless a scope decision changes that recipe.
Preserve its useful output and label it accordingly; do not silently freeze
it as the spec's final tokenizer. Exact budget bytes remove the GB/GiB ambiguity.

Receipt requested in 911c773: code and dependency versions, ordered source
manifest/content identifiers, seed, split algorithm/fraction, actual sampled
UTF-8 bytes and document count, final vocabulary size and tokenizer hash.
The spec's assertion that this sample reaches V remains an expectation until
measured. This review does not certify the real corpus or its run results.

## Positive evidence and limits

The existing six corpus tests cover complete documents, same-process order,
split coverage/disjointness, supplied-tokenizer output, a whole-document byte
budget and malformed input. They pass within the 74-test full gate. The
independent subprocess probe establishes stable order across hash seeds and
confirms row-group contiguity. It does not prove uniform global shuffle,
replay exclusion, training quality or correctness of uninspected run metadata.

## Outcome, appended 2026-10-15 by the other maintainer

Applied on dev/data in the commit that adds this file. Readers are opened for one metadata
pass per file and once per row-group read, each closed again; a test writes 80 files and reads
them all in a subprocess under a hard limit of 64 descriptors. The shuffle is specified as a
block shuffle over row groups with its limits (rows of one group stay contiguous; two documents
of one group are never further apart than the group size) and tested as such, replacing the
wrong contiguity comment. The cross-process claim is now tested across processes with three
PYTHONHASHSEED values. The hash ordering is documented as a deliberate reproducibility choice,
with a seeded generator named as an equally correct alternative. The working-set statement
names what grows with the corpus. The spec's sentence about the 10 GB sample reaching V is an
expectation to be confirmed by the frozen tokenizer's receipt. The running tokenizer is a
web-only pilot, not the final tokenizer: the spec's sample is of the mix, and the code and
Wikipedia/books components are not on the box yet (scope decision recorded in the channel).
