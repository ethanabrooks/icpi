import pickle

from agent.train import make_env
from dollar_lambda import argument, command
from envs.base_env import Env, TimeStep
from tqdm import tqdm


def collect_trajectory(env: Env):
    trajectory = []
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        trajectory.append(TimeStep(state, action, reward, next_state, done))
        state = next_state
    return trajectory


@command(parsers=dict(env_id=argument("env_id")))
def main(
    env_id: str,
    num_trajectories: int,
    seed: int = 0,
    save_path: str = "logs/trajectories.pkl",
):
    env = make_env(env_id, gamma=1.0, seed=seed)
    trajectories = [collect_trajectory(env) for _ in tqdm(range(num_trajectories))]

    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    main()
