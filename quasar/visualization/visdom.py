import numpy as np

import plotly.graph_objs as go


def create_plot_window(vis, x_label, y_label, title, active_trace=None, replace=False):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]),
                    update='replace' if replace else None,
                    name=active_trace,
                    opts=dict(xlabel=x_label, ylabel=y_label, title=title, showlegend=False))


def parallel_coordinates_window(vis, dimensions, title, line_color='blue', win=None):
    data = go.Parcoords(line=dict(color=line_color), dimensions=dimensions)
    return vis._send({'data': [data], 'layout': dict(title=title), 'win': win})
