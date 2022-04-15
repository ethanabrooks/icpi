import pickle

import altair as alt
import pandas as pd

with open("logs/0.pkl", "rb") as f:
    df0 = pickle.load(f)
with open("logs/1.pkl", "rb") as f:
    df1 = pickle.load(f)
with open("logs/2.pkl", "rb") as f:
    df2 = pickle.load(f)

for df in [df0, df1, df2]:
    df.set_index("episode", inplace=True)
df = pd.concat([df0, df1, df2], axis=1)

std = df.std(axis=1)
mean = df.mean(axis=1)
source = pd.DataFrame(dict(regret=mean, lower=mean - std, upper=mean + std))
source.reset_index(inplace=True)

line = (
    alt.Chart(source)
    .mark_line()
    .encode(
        x="episode",
        y=alt.Y(
            "regret", scale=alt.Scale(domain=(0, 1)), axis=alt.Axis(title="regret")
        ),
    )
)

band = (
    alt.Chart(source)
    .mark_area(opacity=0.5)
    .encode(
        x="episode",
        y="lower",
        y2="upper",
    )
)

(band + line).save("regret.html")
