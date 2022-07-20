from typing import Optional


def spec(
    x: str,
    y: str,
    visualizer_url: Optional[str] = None,
    color: str = "run ID",
    scale_type: str = "linear",
) -> dict:
    def subfigure(parameters, x_kwargs=None, y_kwargs=None):
        if y_kwargs is None:
            y_kwargs = {}
        if x_kwargs is None:
            x_kwargs = {}
        return {
            "height": 400,
            "width": 600,
            "encoding": {
                "x": {"type": "quantitative", "field": x, **x_kwargs},
                "y": {"type": "quantitative", "field": y, **y_kwargs},
                "color": {"type": "nominal", "field": color},
                "href": {"field": "url", "type": "nominal"},
                "opacity": {
                    "value": 0.1,
                    "condition": {
                        "test": {
                            "and": [
                                {"param": "legend_selection"},
                                {"param": "hover"},
                            ]
                        },
                        "value": 1,
                    },
                },
            },
            "layer": [
                {
                    "mark": "line",
                    "params": parameters,
                    **(
                        {}
                        if visualizer_url is None
                        else {
                            "transform": [
                                {
                                    "calculate": f"'{visualizer_url}' + datum['run ID']",
                                    "as": "url",
                                }
                            ],
                        }
                    ),
                }
            ],
        }

    params = [
        {
            "bind": "legend",
            "name": "legend_selection",
            "select": {
                "on": "mouseover",
                "type": "point",
                "fields": [color],
            },
        },
        {
            "bind": "legend",
            "name": "hover",
            "select": {
                "on": "mouseover",
                "type": "point",
                "fields": [color],
            },
        },
    ]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "data"},
        "transform": [{"filter": {"field": y, "valid": True}}],
        "hconcat": [
            subfigure(
                parameters=[*params, {"name": "selection", "select": "interval"}],
                x_kwargs={},
                y_kwargs={"scale": {"type": scale_type}},
            ),
            subfigure(
                parameters=params,
                x_kwargs={"scale": {"domain": {"param": "selection", "encoding": "x"}}},
                y_kwargs={
                    "scale": {
                        "type": scale_type,
                        "domain": {"param": "selection", "encoding": "y"},
                    }
                },
            ),
        ],
    }
