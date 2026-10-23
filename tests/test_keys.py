import hashlib
from pathlib import Path

import pytest

from conftest import replaced_after_first_read
from vesper.keys import SequenceDigest, hashed_bytes, verified_bytes


def test_hashed_bytes_returns_the_bytes_it_hashed(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    _ = path.write_bytes(b'{"a": 1}')
    with replaced_after_first_read(path, b'{"a": 2}') as opens:
        data, digest = hashed_bytes(path)
        assert data == b'{"a": 1}'
        assert digest == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert len(opens) == 1  # one read: the hash and the bytes describe one snapshot


def test_verified_bytes_refuses_a_missing_file_and_another_hash(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    _ = path.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    assert verified_bytes(path, digest, "the payload") == b"payload"
    with pytest.raises(ValueError, match=r"the payload .*hash differs"):
        _ = verified_bytes(path, hashlib.sha256(b"other").hexdigest(), "the payload")
    with pytest.raises(ValueError, match="hash differs"):
        _ = verified_bytes(path, None, "the payload")  # a manifest without the entry
    with pytest.raises(ValueError, match="missing"):
        _ = verified_bytes(tmp_path / "absent", digest, "the payload")
    with replaced_after_first_read(path, b"swapped") as opens:
        assert verified_bytes(path, digest, "the payload") == b"payload"
    assert len(opens) == 1


def test_a_sequence_digest_is_framed_and_ordered() -> None:
    a = SequenceDigest()
    a.add(b"ab")
    a.add(b"c")
    b = SequenceDigest()
    b.add(b"a")
    b.add(b"bc")
    assert a.hexdigest() != b.hexdigest()  # the same bytes, another framing
    assert a.count == b.count == 2
    c = SequenceDigest()
    c.add(b"c")
    c.add(b"ab")
    assert c.hexdigest() != a.hexdigest()  # the same items, another order
    d = SequenceDigest()
    d.add(b"ab")
    d.add(b"c")
    assert d.hexdigest() == a.hexdigest()
    assert SequenceDigest().hexdigest() == "0" * 64
    assert len(a.hexdigest()) == 64


def test_a_sequence_digest_continues_from_its_stored_state() -> None:
    whole = SequenceDigest()
    for item in (b"x", b"yy", b"zzz"):
        whole.add(item)
    first = SequenceDigest()
    first.add(b"x")
    resumed = SequenceDigest(first.hexdigest(), first.count)
    resumed.add(b"yy")
    resumed.add(b"zzz")
    assert resumed.hexdigest() == whole.hexdigest()
    assert resumed.count == 3
    with pytest.raises(ValueError, match="64 hex"):
        _ = SequenceDigest("abc")
    with pytest.raises(ValueError, match="64 hex"):
        _ = SequenceDigest("0" * 64, -1)
