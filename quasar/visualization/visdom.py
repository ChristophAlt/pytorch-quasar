import numpy as np

import plotly.graph_objs as go


def create_plot_window(vis, x_label, y_label, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]),
                    opts=dict(xlabel=x_label, ylabel=y_label, title=title))


def parallel_coordinates_window(vis, dimensions, title, line_color='blue'):
    data = go.Parcoords(line=dict(color=line_color), dimensions=dimensions)
    return vis._send({'data': [data], 'layout': dict(title=title)})
