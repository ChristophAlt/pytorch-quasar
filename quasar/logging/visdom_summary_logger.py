import visdom

import numpy as np

import plotly.figure_factory as ff

from itertools import chain

from ..visualization.visdom import create_plot_window, parallel_coordinates_window
from ..visualization.utils import trials_to_dimensions

from .summary_logger import RunSummaryLogger


def to_trials_table(trials, precision=6, sort_by_loss=False):
    fmt = '{{:.{}g}}'
    fmt = fmt.format(precision)

    hparams = list(sorted(set(chain.from_iterable([t.keys() for t in trials]))))
    hparams.remove('loss')
    hparams.remove('run')
    hparams = ['run'] + hparams + ['loss']

    data_matrix = []

    # add header
    data_matrix.append(hparams)

    if sort_by_loss:
        trials = sorted(trials, key=lambda trial: trial['loss'])

    for trial in trials:
        row = []
        for param in hparams:
            value = trial.get(param, float('nan'))
            if isinstance(value, float):
                value = fmt.format(value)
            row.append(value)
        data_matrix.append(row)

    table = ff.create_table(data_matrix)
    return table


class VisdomRunSummaryLogger(RunSummaryLogger):

    def __init__(self, batch_log_interval=100, epoch_log_interval=1,
                 clear_batch_summary=True, **kwargs):
        super(VisdomRunSummaryLogger, self).__init__()
        self.batch_log_interval = batch_log_interval
        self.epoch_log_interval = epoch_log_interval
        self.clear_batch_summary = clear_batch_summary

        use_incoming_socket = kwargs.pop('use_incoming_socket', False)

        self.visdom = visdom.Visdom(use_incoming_socket=use_incoming_socket, **kwargs)
        self.windows = {}
        self.run_params = []

    def on_run_start(self, trainer):
        if self.window_exists('train_loss') and self.clear_batch_summary:
            self.clear_window('train_loss')

    def batch_summary(self, trainer):
        metric = 'train_loss'

        if not self.window_exists(metric):
            self.create_window(metric, x_label='iteration', y_label='loss',
                               trace=None)

        iteration = trainer.state.iteration - 1
        if iteration % self.batch_log_interval == 0:
            self.log_metric(metric,
                            x=trainer.state.iteration,
                            y=trainer.state.output,
                            trace=None,
                            show_legend=False)

    def epoch_summary(self, evaluator, trainer, phase):
        for name, value in evaluator.state.metrics.items():
            metric = '{}_{}'.format(phase, name)

            if not self.window_exists(metric):
                self.create_window(metric, x_label='epoch', y_label=name,
                                   trace=self.run_name)

            self.log_metric(metric, x=trainer.state.epoch, y=value,
                            trace=self.run_name)

    def run_summary(self, params, loss):
        params_copy = dict(params)
        params_copy['loss'] = loss
        params_copy['run'] = self.run_name
        self.run_params.append(params_copy)

        figure = to_trials_table(self.run_params, sort_by_loss=True)
        if 'params_table' not in self.windows:
            self.windows['params_table'] = self.visdom._send({
                'data': figure['data'],
                'layout': figure['layout']
            })

        self.visdom._send({
            'data': figure['data'],
            'layout': figure['layout'],
            'win': self.windows['params_table']
        })

        if 'params_pcoo' not in self.windows:
            self.windows['params_pcoo'] = parallel_coordinates_window(
                self.visdom, dimensions=trials_to_dimensions(self.run_params),
                title='Hyperparameters')

        parallel_coordinates_window(self.visdom,
                                    dimensions=trials_to_dimensions(self.run_params),
                                    title='Hyperparameters',
                                    win=self.windows['params_pcoo'])

    def window_exists(self, metric):
        return metric in self.windows

    def clear_window(self, metric, trace=None):
        self.visdom.line(X=np.array([1]), Y=np.array([np.nan]),
                         update='replace',
                         name=trace,
                         win=self.windows[metric])

    def create_window(self, metric, x_label, y_label, trace=None):
        if not self.window_exists(metric):
            self.windows[metric] = create_plot_window(self.visdom,
                                                      x_label=x_label,
                                                      y_label=y_label,
                                                      active_trace=trace,
                                                      title=metric)

    def log_metric(self, metric, x, y, trace=None, show_legend=True):
        self.visdom.line(X=np.array([x]),
                         Y=np.array([y]),
                         name=trace,
                         update='append',
                         win=self.windows[metric],
                         opts=dict(showlegend=show_legend))
