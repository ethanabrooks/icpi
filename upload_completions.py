import os
import shelve

from dollar_lambda import command
from run_logger import HasuraLogger
from tqdm import tqdm

from gql import gql


@command()
def main(
    temperature: float = 0.1,
    top_p: float = 1.0,
):
    logger = HasuraLogger(os.getenv("GRAPHQL_ENDPOINT"))
    with shelve.open("completions/completions.pkl") as completions:
        for k, v in tqdm(completions.items(), total=len(completions)):
            logger.execute(
                query=gql(
                    """
mutation post_completion($prompt: String!, $completion: String!, $temperature: numeric!, $top_p: numeric!, $max_tokens: Int) {
  insert_completions_one(object: {completion: $completion, prompt: $prompt, temperature: $temperature, top_p: $top_p, max_tokens: $max_tokens}, on_conflict: {constraint: completions_pkey, update_columns: completion}) {
    completion
  }
}
"""
                ),
                variable_values=dict(
                    prompt=k,
                    completion=v,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=None,
                ),
            )


if __name__ == "__main__":
    main()
