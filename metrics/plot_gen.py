from typing import List, Tuple, Callable

import numpy as np
import plotly.graph_objects as go


def create_fig(x_y_ths: List[Tuple[np.ndarray, ...]],
               class_list: List[str],
               axis_labels: Tuple[str, str],
               metric_name: str,
               metric_func: Callable,
               line_mode: int) -> go.Figure:
    fig = go.Figure()

    y0 = 0 if line_mode == 0 else 1
    y1 = 1 - y0
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, y0=y0, x1=1, y1=y1
    )

    for i, (x_y_th, cls) in enumerate(zip(x_y_ths, class_list)):
        x, y, th = x_y_th
        cls = cls.replace('_', ' ')
        cls = f'{i}.{cls}'

        thresholds = [f'threshold: {th_s:.5f}' for th_s in th]
        fig.add_trace(go.Scatter(x=x, y=y, text=thresholds,
                                 name=f'{cls:20} {metric_name}: {metric_func(x, y):.3f}', mode='lines'))

    fig.update_layout(
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=800, height=800,
        font=dict(family='Courier New', size=10)
    )
    return fig
