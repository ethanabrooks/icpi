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
from rl.train import train
from run_logger import HasuraLogger
from run_logger.main import get_config_params
from vega_charts import line

tree = CommandTree()

DEFAULT_CONFIG = "config.yml"
GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT")


@tree.command()
def no_logging(
    config: str = DEFAULT_CONFIG,
    debug: int = 0,
    load_id: Optional[int] = None,
    require_cache: bool = False,
):
    logger = HasuraLogger(GRAPHQL_ENDPOINT)
    params = dict(get_config_params(config), debug=debug)
    if load_id is not None:
        params.update(run_logger.get_load_params(load_id=load_id, logger=logger))
    train(**params, logger=logger, require_cache=require_cache)


@tree.subcommand(parsers=dict(name=argument("name")))
def log(
    name: str,
    allow_dirty: bool = False,
    config: str = DEFAULT_CONFIG,
    require_cache: bool = False,
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
        line.spec(x=x, y=y, visualizer_url=visualizer_url)
        for x in ["step", "hours"]
        for y in ["regret", "return", "use_model_prob", "eval regret", "eval return"]
    ]

    params, logger = run_logger.initialize(
        graphql_endpoint=GRAPHQL_ENDPOINT,
        config=config,
        charts=charts,
        metadata=metadata,
        name=name if sweep_id is None else None,
        load_id=None,
        sweep_id=sweep_id,
    )
    train(**params, debug=0, logger=logger, require_cache=require_cache)


if __name__ == "__main__":
    tree()
