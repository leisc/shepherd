"""Demo emitters for the first-run Shepherd tour."""

from __future__ import annotations

import importlib.resources

import click

_DEMOS = {
    "offline-task": "offline_task.py",
    "quickstart": "world_channel.py",
    "world-channel": "world_channel.py",
    "agent-task": "agent_task.py",
    "claude-readme": "claude_readme.py",
}


@click.group()
def demo() -> None:
    """Emit checked-in quickstart demo scripts."""


@demo.command("write")
@click.argument("name", type=click.Choice(sorted(_DEMOS)))
def demo_write(name: str) -> None:
    """Write demo NAME to standard output."""
    templates = importlib.resources.files("shepherd.templates.quickstart")
    click.echo(templates.joinpath(_DEMOS[name]).read_text(encoding="utf-8"), nl=False)


__all__ = ["demo"]
