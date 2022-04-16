import os
import socket
import sys
import time
from pathlib import Path
from shlex import quote
from typing import Optional

import run_logger
from dollar_lambda import CommandTree, argument
from git import Repo
from train import train
from vega_charts import line

tree = CommandTree()

DEFAULT_CONFIG = "config.yml"


@tree.command()
def run_with_config(
    config: str = DEFAULT_CONFIG,
    debug: bool = False,
):
    params, logger = run_logger.initialize(config=config)
    params.update(debug=debug)
    train(**params, logger=logger)


@tree.subcommand(parsers=dict(name=argument("name")))
def log(
    name: str,
    allow_dirty: bool = False,
    config: str = DEFAULT_CONFIG,
    load_id: Optional[int] = None,
    sweep_id: Optional[int] = None,
):
    repo = Repo(".")
    if not allow_dirty:
        assert not repo.is_dirty()
    metadata = dict(
        reproducibility=(
            dict(
                command_line=f'python {" ".join(quote(arg) for arg in sys.argv)}',
                time=time.strftime("%c"),
                cwd=str(Path.cwd()),
                commit=str(repo.commit()),
                remotes=[*repo.remote().urls],
            )
        ),
        hostname=socket.gethostname(),
    )

    visualizer_url = os.getenv("VISUALIZER_URL")
    assert visualizer_url is not None, "VISUALIZER_URL must be set"
    charts = [
        line.spec(x="step", y=y, visualizer_url=visualizer_url)
        for y in ["regret", "return"]
    ]

    params, logger = run_logger.initialize(
        graphql_endpoint=os.getenv("GRAPHQL_ENDPOINT"),
        config=config,
        charts=charts,
        metadata=metadata,
        name=name,
        load_id=load_id,
        sweep_id=sweep_id,
    )
    train(**params, debug=False, logger=logger)


if __name__ == "__main__":
    tree()
