import numpy as np
from dollar_lambda import command


@command()
def main(
    std: float,
    mean: float = 2.5,
    min_w: float = 2,
    max_w: float = 4,
    n_w: int = 20,
    test_time: float = 3600,
):
    rng = np.random.default_rng(0)

    for w in np.linspace(min_w, max_w, n_w + 1):
        t = 0
        requests = []
        successes = 0
        while t < test_time:
            s = rng.normal(mean, std)
            t += w
            requests.append(t + s)
            if len(requests) < 20:
                successes += 1
            requests = [r for r in requests if r >= t - 60]
        print(f"{w:.2f}: {successes}")


if __name__ == "__main__":
    main()
