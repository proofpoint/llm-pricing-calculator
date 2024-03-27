import subprocess
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command("run")
@click.option("--port", "-p", default="8000", help="Port to run the calculator on.")
@click.option("--debug", "-d", is_flag=True, default=False, help="Run in debug mode.")
def run(port: str, debug: bool):
    this_file = Path(__file__)

    args = [
        "streamlit",
        "run",
        str(this_file.parent / "main_streamlit.py"),
        "--server.port",
        str(port),
    ]
    if debug:
        args.append("--global.developmentMode=TRUE")
    subprocess.run(args)
