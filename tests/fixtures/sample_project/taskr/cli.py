import click


@click.group()
def cli():
    """taskr command line interface."""


@cli.command()
@click.argument("title")
@click.option("--priority", default=1)
def add(title: str, priority: int):
    """Add a task from the command line."""
    click.echo(f"added {title} ({priority})")
