# Precise local stubs for the parts of `pyarrow` this project uses (vesper/corpus.py,
# vesper/collect.py and their tests). The library ships no complete hints; these declare only
# what is called, exactly.

from collections.abc import Mapping, Sequence

from pyarrow import parquet as parquet

class DataType: ...

def string() -> DataType: ...

class Field: ...

def field(name: str, data_type: DataType) -> Field: ...

class Schema:
    names: list[str]

def schema(fields: Sequence[Field]) -> Schema: ...

class ChunkedArray:
    def to_pylist(self) -> list[object]: ...

class Table:
    num_rows: int
    schema: Schema
    def column(self, i: int | str) -> ChunkedArray: ...
    @staticmethod
    def from_pydict(
        mapping: Mapping[str, Sequence[object]], schema: Schema | None = ...
    ) -> Table: ...
