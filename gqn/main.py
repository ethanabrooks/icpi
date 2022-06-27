import os
import socket
import sys
import time
from pathlib import Path
from shlex import quote
from typing import Optional

from dollar_lambda import CommandTree, argument, flag, nonpositional, option
from git import Repo
from rl.baseline import deep_baseline, tabular_main
from rl.train import train
from run_logger import HasuraLogger
from run_logger.main import get_config_params, get_load_params
from vega_charts import line

tree = CommandTree()

DEFAULT_CONFIG = "config.yml"
GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT")


def validate_local_rank(s: str):
    assert "--local-rank=" in s
    return s


def main(model_name: str, seed: "int | list[int]", **kwargs):
    if model_name.startswith("baseline"):
        train_fn = deep_baseline
    elif model_name == "tabular-q":
        train_fn = tabular_main
    else:
        train_fn = train
    kwargs.update(model_name=model_name, seed=seed)
    if isinstance(seed, list):
        seeds = list(seed)
        for seed in seeds:
            kwargs.update(seed=seed)
            train_fn(**kwargs)
    else:
        train_fn(**kwargs)


ALLOW_DIRTY_FLAG = flag("allow_dirty", default=False)  # must be set from CLI
LOCAL_RANK_ARG = argument("local_rank", type=validate_local_rank).optional().ignore()
REQUIRE_CACHE_FLAG = flag("require_cache", default=False)  # must be set from CLI


@tree.command(
    parsers=dict(
        kwargs=nonpositional(
            option("debug", type=int, default=0),
            REQUIRE_CACHE_FLAG,
            option("t_threshold", type=int, default=None),
            LOCAL_RANK_ARG,
        )
    )
)
def no_logging(
    config: str = DEFAULT_CONFIG,
    load_id: Optional[int] = None,
    **kwargs,
):
    logger = HasuraLogger(GRAPHQL_ENDPOINT)
    params = get_config_params(config)
    if load_id is not None:
        load_params = get_load_params(load_id=load_id, logger=logger)
        params.update(load_params)  # load params > config params
    params.update(kwargs)  # kwargs params > load params > config params
    main(logger=logger, **params)


@tree.subcommand(
    parsers=dict(
        name=argument("name"),
        kwargs=nonpositional(
            LOCAL_RANK_ARG,
            REQUIRE_CACHE_FLAG,
        ),
    )
)
def logging(
    allow_dirty: bool,
    name: str,
    repo: Repo,
    sweep_id: Optional[int],
    **kwargs,
):
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
    ] + [line.spec(x="hours", y="seconds per query", visualizer_url=visualizer_url)]

    logger = HasuraLogger(GRAPHQL_ENDPOINT)
    sweep_params = logger.create_run(
        metadata=metadata, sweep_id=sweep_id, charts=charts
    )
    kwargs.update(sweep_params)  # sweep params > config params
    logger.update_metadata(  # this updates the metadata stored in the database
        dict(parameters=kwargs, run_id=logger.run_id, name=name)
    )  # todo: encapsulate in HasuraLogger
    main(**kwargs)
