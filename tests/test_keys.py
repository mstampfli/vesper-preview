import pytest

from vesper.keys import SequenceDigest


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
