import time
from typing import Optional

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CallbackList
from run_logger import HasuraLogger

from base_env import Env
from rl.common import make_env, make_log, evaluate


class LoggingCallback(BaseCallback):
    def __init__(self, logger: HasuraLogger, start_time: float, verbose=0):
        super().__init__(verbose)
        self.run_logger=logger
        self.start_time = start_time

    def _on_step(self):
        if np.sum(self.locals["dones"]) > 0:
            assert len(self.locals["dones"]) == 1
            make_log(
                logger=self.run_logger,
                info=self.locals["infos"][0],
                rewards=self.locals["rewards"].tolist(),
                evaluation=False,
                use_model_prob=0.0,
                success_buffer_size=0,
                gamma=self.training_env.envs[0].gamma(),
                start_time=self.start_time,
                step=self.model.num_timesteps,
            )


class EvalCallback(BaseCallback):
    def __init__(self,
                 logger: HasuraLogger,
                 eval_env: Env,
                 eval_interval: int,
                 start_time: float,
                 verbose=0):
        super().__init__(verbose)
        self.run_logger = logger
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.start_time = start_time

    def _on_step(self):
        evaluate(
            logger=self.run_logger,
            env=self.eval_env,
            eval_interval=self.eval_interval,
            act_fn=self.act,
            use_model_prob=0.0,
            success_buffer_size=0,
            gamma=self.eval_env.gamma(),
            start_time=self.start_time,
            step=self.model.num_timesteps,
        ) 

    def act(self, _, state):
        return self.model.predict(np.array(state))[0].item()


def deep_baseline(
    logger: HasuraLogger,
    model_name: str,
    env_id: str,
    total_steps: int,
    eval_interval: Optional[int],
    seed: int,
    **_):

    train_env = make_env(env_id, seed, False) 
    eval_env = make_env(env_id, seed, False) 
    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[3,3]),
        gamma=train_env.gamma(),
        seed=seed,
        verbose=0
    )

    start_time = time.time()
    eval_callback = EvalCallback(
        logger,
        eval_env,
        eval_interval,
        start_time
    )
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([
            LoggingCallback(logger, start_time=start_time),
            EveryNTimesteps(n_steps=eval_interval, callback=eval_callback),
        ]),
    )
