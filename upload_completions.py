import os
import shelve

from dollar_lambda import command
from run_logger import HasuraLogger
from tqdm import tqdm

from gqn.gpt3 import post_completion


@command()
def main(
    temperature: float = 0.1,
    top_p: float = 1.0,
):
    logger = HasuraLogger(os.getenv("GRAPHQL_ENDPOINT"))
    with shelve.open("completions/completions.pkl") as completions:
        for k, v in tqdm(completions.items(), total=len(completions)):
            post_completion(
                logger=logger,
                prompt=k,
                completion=v,
                temperature=temperature,
                top_p=top_p,
            )


if __name__ == "__main__":
    main()
