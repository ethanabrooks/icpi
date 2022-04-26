import compare_features
import numpy as np
from compare_features import Encoder
from dollar_lambda import command

ACTIONS = ["Left", "Stay", "Right"]


class PaddleXBallXParensBallY(Encoder):
    def name(self) -> str:
        return "{paddle_x},{ball_x} ({ball_y} to go). Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"{paddle_x},{ball_x} ({ball_y} to go)."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, _ = next_state
        return f"{paddle_x},{ball_x} ({'success' if reward == 1 else 'failure'})."


class ParensPaddleParensBall(Encoder):
    def name(self) -> str:
        return "({paddle_x},0) ({ball_x},{ball_y}). Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"({paddle_x},0) ({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"({paddle_x},0) ({ball_x},{ball_y}) [{'success' if reward == 1 else 'failure'}]."


class ParensPaddleParensBallWithNames(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}). Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


class ParensPaddleParensBallWithNamesAndVerboseActions(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) Move paddle right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{'Do not move paddle' if action == 1 else ('Move paddle ' + ACTIONS[action].lower())}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


class ParensPaddleParensBallWithNamesAndFalling(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [falling]. Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [falling]."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}."

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


class ParensPaddleParensBallWithNamesAndCanCatch(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [can catch the ball]. Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        can_catch = ball_y >= abs(paddle_x - ball_x)
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'can catch the ball' if can_catch else 'cannot catch the ball'}]."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


@command()
def main(
    actions_path: str,
    transitions_path: str,
    n: int = 40,
    seed: int = 0,
    logprobs: int = 3,
):
    compare_features.main(
        actions_path=actions_path,
        transitions_path=transitions_path,
        n=n,
        seed=seed,
        logprobs=logprobs,
        encoders=[
            PaddleXBallXParensBallY(),
            ParensPaddleParensBall(),
            ParensPaddleParensBallWithNames(),
            ParensPaddleParensBallWithNamesAndVerboseActions(),
            ParensPaddleParensBallWithNamesAndFalling(),
            ParensPaddleParensBallWithNamesAndCanCatch(),
        ],
    )


if __name__ == "__main__":
    main()
