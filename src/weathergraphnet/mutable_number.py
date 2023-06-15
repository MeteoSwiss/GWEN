"""Mutable number."""
# Standard library
from typing import List


class MutableNumber:
    """A mutable number."""

    def __init__(self, number: float) -> None:
        """Create an instance of ``MutableNumber``.

        Args:
            number: Initial number.

        """
        self.history: List[float] = [float(number)]

    def get(self, idx: int = -1) -> float:
        """Get the current or a past value of the number.

        Args:
            idx (optional): Index since the initial value. Defaults to the most
                recent (i.e., current) value.

        """
        return self.history[idx]

    def add(self, addend: float) -> "MutableNumber":
        """Add ``addend`` to the current number."""
        number = self.get() + float(addend)
        self.history.append(number)
        return self

    def subtract(self, subtrahend: float) -> "MutableNumber":
        """Subtract ``subtrahend`` from the current number."""
        number = self.get() - float(subtrahend)
        self.history.append(number)
        return self

    def multiply(self, factor: float) -> "MutableNumber":
        """Multiply current number by ``factor``."""
        number = self.get() * float(factor)
        self.history.append(number)
        return self

    def divide(self, divisor: float) -> "MutableNumber":
        """Divide current number by ``divisor``."""
        number = self.get() / float(divisor)
        self.history.append(number)
        return self
