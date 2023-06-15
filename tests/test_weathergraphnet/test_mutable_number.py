"""Tests for `weathergraphnet` package."""
# First-party
from weathergraphnet.mutable_number import MutableNumber


def test_get():
    n = MutableNumber(3)
    assert n.get() == 3


def test_add():
    n = MutableNumber(4).add(2)
    assert n.get() == 6
    assert n.history == [4, 6]


def test_subtract():
    n = MutableNumber(4).subtract(2)
    assert n.get() == 2
    assert n.history == [4, 2]


def test_multiply():
    n = MutableNumber(4).multiply(2)
    assert n.get() == 8
    assert n.history == [4, 8]


def test_divide():
    n = MutableNumber(4).divide(2)
    assert n.get() == 2
    assert n.history == [4, 2]


def test_chain():
    n = MutableNumber(4).add(1).subtract(2).multiply(3).divide(9)
    assert n.get() == 1
    assert n.history == [4, 5, 3, 9, 1]
