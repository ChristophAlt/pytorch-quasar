import numpy as np


def create_plot_window(vis, x_label, y_label, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]),
                    opts=dict(xlabel=x_label, ylabel=y_label, title=title))
