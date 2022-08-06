"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import re
from typing import Iterable, NamedTuple, Optional, SupportsFloat, Tuple

import base_env
import gym
import numpy as np
from base_env import ActType, ObsType, TimeStep
from gym import logger, spaces
from gym.error import DependencyNotInstalled
from rl.lm import Data


def verify_number_and_cast(x: SupportsFloat) -> float:
    """Verify parameter is a single number and cast to a float."""
    try:
        x = float(x)
    except (ValueError, TypeError):
        raise ValueError(f"An option ({x}) could not be converted to a float.")
    return x


def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Tuple[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.
    Args:
      options: Options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.
    Returns:
      Tuple of the lower and upper limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high

    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) must be lower than higher bound ({high})."
        )

    return low, high


class CartPoleEnv(gym.Env):
    """
    ### Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.
    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    ### Arguments
    ```
    gym.make('CartPole-v1')
    ```
    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, random_seed: int, render_mode: Optional[str] = None):
        self.np_random = np.random.default_rng(seed=random_seed)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.tau *= 2
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2, seed=random_seed)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        # self.renderer = Renderer(self.render_mode, self._render)

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        # self.renderer.render_step()
        return np.array(self.state, dtype=np.float32), reward, terminated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # super().reset()
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        # self.renderer.reset()
        # self.renderer.render_step()
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        if self.render_mode is not None:
            raise NotImplementedError
            # return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            raise NotImplementedError
            # import pygame
            #
            # pygame.display.quit()
            # pygame.quit()
            # self.isopen = False


REWARDS = {0.0: "Failure", 1.0: "Success"}


class Obs(NamedTuple):
    x: float
    x_dot: float
    theta: float
    theta_dot: float

    def __str__(self):
        return str(tuple([round(x, 3) for x in self]))
        # x, x_dot, theta, theta_dot = [f"{round(x, 3):.3f}" for x in self]
        # return f"S(x={x}, x_dot={x_dot}, theta={theta}, theta_dot={theta_dot})"


class Wrapper(gym.Wrapper, base_env.Env[Obs, int]):
    def __init__(self, *args, hint: bool, **kwargs):
        self.hint = hint
        self.data = Data.code
        super().__init__(*args, **kwargs)

    @staticmethod
    def action_stop() -> str:
        return "\n"

    def action_str(self, action: ActType) -> str:
        action_str = self.actions()[action]
        return f"s = {action_str}(s){self.action_stop()}"

    def actions(self) -> "list[str]":
        return ["left", "right"]

    @staticmethod
    def done_stop() -> str:
        return "\n"

    def done_str(self, done: bool) -> str:
        return f"assert{' ' if done else ' not '}done"

    @staticmethod
    def gamma() -> float:
        return 0.8

    def hint_str(self, state: Obs) -> str:
        x, x_dot, theta, theta_dot = state
        x_threshold = self.x_threshold
        x_threshold_str = f"{round(x_threshold, 3):.3f}"
        x_str = f"{round(x, 3):.3f}"
        if x < -x_threshold:
            x_hint = f"{x_str} < -{x_threshold_str}"
        elif x_threshold < x:
            x_hint = f"{x_threshold_str} < {x_str}"
        elif -x_threshold <= x <= x_threshold:
            x_hint = f"-{x_threshold_str} <= {x_str} <= {x_threshold_str}"
        else:
            raise RuntimeError()
        theta_threshold = self.theta_threshold_radians
        theta_threshold_str = f"{round(theta_threshold, 3):.3f}"
        theta_str = f"{round(theta, 3):.3f}"
        if theta < -theta_threshold:
            theta_hint = f"{theta_str} < -{theta_threshold_str}"
        elif theta_threshold < theta:
            theta_hint = f"{theta_threshold_str} < {theta_str}"
        elif -theta_threshold <= theta <= theta_threshold:
            theta_hint = (
                f"-{theta_threshold_str} <= {theta_str} <= {theta_threshold_str}"
            )
        else:
            raise RuntimeError()
        hint = " and ".join([x_hint, theta_hint])
        return hint

    @staticmethod
    def initial_str() -> str:
        return "\ns = reset()\n"

    def max_q_steps(self) -> int:
        return 200

    def reward_str(self, reward: float) -> str:
        return f"assert reward == {int(reward)}"

    def start_states(self) -> Optional[Iterable[ObsType]]:
        pass

    def state_str(self, state: ObsType) -> str:
        state_str = f"assert s == {state}"
        hint_str = self.hint_str(state)
        if self.hint and hint_str:
            state_str += f" and {hint_str}"
        return state_str + self.state_stop()

    def valid_done(self, done_str: str) -> bool:
        return (
            done_str.startswith("assert")
            and "done" in done_str
            and done_str.endswith(self.done_stop())
        )

    def valid_reward(self, reward_str: str) -> bool:
        return bool(
            re.findall(r"assert reward == \d+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(state_str.startswith("assert s == ")) and state_str.endswith(
            self.state_stop()
        )

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def partially_observable() -> bool:
        return False

    def quantify(self, prompt: str, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = self.gamma()
        matches = re.findall(r"reward == (\d)", prompt)
        matches = matches[: self.max_q_steps()]
        return sum([gamma**t * float(x) for t, x in enumerate(matches)])

    def reset(self, **kwargs):
        return Obs(*super().reset())

    def step(self, action):
        s, r, t, i = super().step(action)
        return Obs(*s), r, t, i

    def ts_to_string(self, ts: TimeStep) -> str:
        reward_str = f"assert reward == {ts.reward}"
        state_str = self.state_str(ts.state)
        parts = [
            state_str,
            self.action_str(ts.action),
            reward_str,
            self.reward_stop(),
        ]
        if ts.done:
            next_state_str = self.state_str(ts.next_state)
            parts += [next_state_str]
        if self.hint:
            if not ((" < " in state_str) == ts.done):
                breakpoint()
        return "".join(parts)


if __name__ == "__main__":
    env = Wrapper(CartPoleEnv(0), hint=True)
    d = True
    t = 0
    while True:
        if d:
            s = env.reset()
        a = env.action_space.sample()
        _s, r, d, i = env.step(a)
        string = env.ts_to_string(TimeStep(s, a, r, d, _s))
        # print(string)
        s = _s
        if d:
            print(t)
            t = 0
        t += 1
