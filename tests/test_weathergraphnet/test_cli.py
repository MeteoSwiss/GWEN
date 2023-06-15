"""Test module ``weathergraphnet``."""
# Third-party
from click.testing import CliRunner

# First-party
from weathergraphnet import cli


class _TestCLI:
    """Base class to test the command line interface."""

    def call(self, args=None, *, exit_=0):
        runner = CliRunner()
        result = runner.invoke(cli.main, args)
        assert result.exit_code == exit_
        return result


class TestNoCmd(_TestCLI):
    """Test CLI without commands."""

    def test_default(self):
        result = self.call()
        assert result.output.startswith("Usage: ")
        assert "Show this message and exit." in result.output

    def test_help(self):
        result = self.call(["--help"])
        assert result.output.startswith("Usage: ")
        assert "Show this message and exit." in result.output

    def test_version(self):
        result = self.call(["-V"])
        assert cli.__version__ in result.output

    def test_only_arg(self):
        result = self.call(["4"])
        assert result.output.strip() == "4"

    def test_two_args(self):
        result = self.call(["4", "5"], exit_=2)
        assert "Error: No such command '5'." in result.output.split("\n")


class TestOneCmd(_TestCLI):
    """Test CLI with single command."""

    def test_add(self):
        result = self.call(["4", "plus", "2"])
        assert result.output.strip() == "6"

    def test_subtract(self):
        result = self.call(["4", "minus", "2"])
        assert result.output.strip() == "2"

    def test_multiply(self):
        result = self.call(["4", "times", "2"])
        assert result.output.strip() == "8"

    def test_divide(self):
        result = self.call(["4", "by", "2"])
        assert result.output.strip() == "2"


class TestMultCmds(_TestCLI):
    """Test CLI with multiple commands."""

    args = ["4", "plus", "1", "minus", "2", "times", "3", "by", "9"]

    def test_result_only(self):
        result = self.call(self.args)
        assert result.output.strip() == "1"

    def test_verbose(self):
        result = self.call(["-v"] + self.args)
        result = result.output.strip().split("\n")
        solution = ["4 + 1 = 5", "5 - 2 = 3", "3 * 3 = 9", "9 / 9 = 1", "1"]
        assert len(result) == len(solution)
        assert result == solution
