import pickle

from agent.train import make_env
from base_env import Env, TimeStep
from dollar_lambda import command


def collect_trajectories(env: Env, num_trajectories: int):
    trajectories = []
    for _ in range(num_trajectories):
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            trajectory.append(TimeStep(state, action, reward, next_state, done))
            state = next_state
        trajectories.append(trajectory)
    return trajectories


@command()
def main(env_id: str, seed: int, num_trajectories: int, save_path: str):
    env = make_env(env_id, gamma=1.0, seed=seed)
    trajectories = collect_trajectories(env, num_trajectories)

    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    main()
