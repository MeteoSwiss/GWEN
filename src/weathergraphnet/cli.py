"""Command line interface of weathergraphnet."""
# Third-party
import click

# Local
from . import __version__
from .mutable_number import MutableNumber


# pylint: disable=W0613  # unused-argument (param)
def print_version(ctx, param, value: bool) -> None:
    """Print the version number and exit."""
    if value:
        click.echo(__version__)
        ctx.exit(0)


# pylint: disable=W0613  # unused-argument (args, kwargs)
@click.pass_context
def print_number(ctx, *args, **kwargs) -> None:
    """Print the current number."""
    number = ctx.obj["number"].get()
    click.echo(f"{number:g}")


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    invoke_without_command=True,
    chain=True,
    result_callback=print_number,
)
@click.argument(
    "number",
    type=float,
    nargs=1,
)
@click.option(
    "--version",
    "-V",
    help="Print version and exit.",
    is_flag=True,
    expose_value=False,
    callback=print_version,
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (specify multiple times for more).",
)
@click.pass_context
def main(ctx, number: float, **kwargs) -> None:
    """Console script for test_cli_project."""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["number"] = MutableNumber(number)
    ctx.obj.update(kwargs)


def print_operation(ctx, operator: str, value: float) -> None:
    if ctx.obj["verbose"]:
        number = ctx.obj["number"]
        click.echo(f"{number.get(-2):g} {operator} {value:g} = {number.get():g}")


@main.command("plus", help="addition")
@click.argument("addend", type=float, nargs=1)
@click.pass_context
def plus(ctx, addend: float) -> None:
    ctx.obj["number"].add(addend)
    print_operation(ctx, "+", addend)


@main.command("minus", help="subtraction")
@click.argument("subtrahend", type=float, nargs=1)
@click.pass_context
def minus(ctx, subtrahend: float) -> None:
    ctx.obj["number"].subtract(subtrahend)
    print_operation(ctx, "-", subtrahend)


@main.command("times", help="multiplication")
@click.argument("factor", type=float, nargs=1)
@click.pass_context
def times(ctx, factor: float) -> None:
    ctx.obj["number"].multiply(factor)
    print_operation(ctx, "*", factor)


@main.command("by", help="division")
@click.argument("divisor", type=float, nargs=1)
@click.pass_context
def by(ctx, divisor: float) -> None:
    ctx.obj["number"].divide(divisor)
    print_operation(ctx, "/", divisor)
