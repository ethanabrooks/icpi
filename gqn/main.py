import os
import socket
import sys
import time
from pathlib import Path
from shlex import quote
from typing import Optional

import ray
import yaml
from dollar_lambda import CommandTree, argument, flag, nonpositional, option
from git import Repo
from ray import tune
from rl.baseline import deep_baseline, tabular_main
from rl.train import train
from run_logger import HasuraLogger
from run_logger.main import get_config_params, get_load_params
from sweep_logger import create_sweep
from sweep_logger.create_sweep import SweepMethod, compute_remaining_runs
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


# must be set from CLI
ALLOW_DIRTY_FLAG = flag("allow_dirty", default=False)
LOCAL_RANK_ARG = option("local_rank").optional().ignore()
NO_CACHE_FLAG = flag("use_cache", default=True, string="--no-cache")
REQUIRE_CACHE_FLAG = flag("require_cache", default=False)


@tree.command(
    parsers=dict(
        kwargs=nonpositional(
            option("debug", type=int, default=0),
            option("t_threshold", type=int, default=None),
            LOCAL_RANK_ARG,
            NO_CACHE_FLAG,
            REQUIRE_CACHE_FLAG,
        )
    )
)
def no_log(
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


def _log(
    allow_dirty: bool,
    eval_interval: Optional[int],
    name: str,
    repo: Repo,
    require_cache: bool,
    sweep_id: Optional[int],
    use_cache: bool,
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

    def xy():
        ys = [
            "regret",
            "return",
            "use_model_prob",
        ]
        if eval_interval is not None:
            ys.extend(["eval regret", "eval return"])
        for y in ys:
            for x in ["step", "hours"]:
                yield x, y
        for y in [
            "API error probability",
            "seconds per query",
            "seconds per completion",
        ]:
            yield "hours", y

    charts = [
        line.spec(color="seed", x=x, y=y, visualizer_url=visualizer_url)
        for x, y in xy()
    ]

    logger = HasuraLogger(GRAPHQL_ENDPOINT)
    logger.create_run(metadata=metadata, sweep_id=sweep_id, charts=charts)
    logger.update_metadata(  # this updates the metadata stored in the database
        dict(parameters=kwargs, run_id=logger.run_id, name=name)
    )  # todo: encapsulate in HasuraLogger
    main(
        **kwargs,
        eval_interval=eval_interval,
        logger=logger,
        require_cache=require_cache,
        use_cache=use_cache,
    )


@tree.subcommand(
    parsers=dict(
        kwargs=nonpositional(
            argument("name"),
            ALLOW_DIRTY_FLAG,
            LOCAL_RANK_ARG,
            NO_CACHE_FLAG,
            REQUIRE_CACHE_FLAG,
        ),
    )
)
def log(config: str = DEFAULT_CONFIG, **kwargs):
    repo = Repo(".")
    params = get_config_params(config)
    params.update(kwargs)
    return _log(**params, debug=0, repo=repo, sweep_id=None)


def trainable(config: dict):
    return _log(**config)


@tree.subcommand(
    parsers=dict(
        name=argument("name"),
        kwargs=nonpositional(
            ALLOW_DIRTY_FLAG,
            LOCAL_RANK_ARG,
            NO_CACHE_FLAG,
            REQUIRE_CACHE_FLAG,
        ),
    )
)
def sweep(
    name: str,
    config: str = DEFAULT_CONFIG,
    random_search: bool = False,
    num_runs: Optional[int] = None,
    **kwargs,
):
    config_path = Path(config)
    with config_path.open() as f:
        config = yaml.load(f, yaml.FullLoader)
    if num_runs is None:
        num_runs = compute_remaining_runs(config)

    config = dict(
        name=name,
        repo=Repo("."),
        debug=-1,  # do not print <>
        **kwargs,
        **{
            k: (tune.choice(v) if random_search else tune.grid_search(v))
            if isinstance(v, list)
            else v
            for k, v in config.items()
        },
    )
    method = SweepMethod.random if random_search else SweepMethod.grid
    sweep_id = create_sweep.run(
        config=config_path,
        graphql_endpoint=GRAPHQL_ENDPOINT,
        log_level="INFO",
        method=method.name,
        name=name,
        project=None,
        remaining_runs=num_runs,
    )
    config.update(sweep_id=sweep_id)
    ray.init()
    analysis = tune.run(trainable, config=config)
    print(analysis.stats())


if __name__ == "__main__":
    tree()
