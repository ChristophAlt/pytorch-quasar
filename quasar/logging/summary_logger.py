import visdom
import numpy as np

from abc import ABCMeta, abstractmethod

from ignite.engine import Events

from quasar.visualization.visdom import create_plot_window


class SummaryLogger(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def on_run_start(self, engine):
        pass

    @abstractmethod
    def log_batch_summary(self, engine):
        pass

    @abstractmethod
    def log_epoch_summary(self, engine, trainer, phase):
        pass

    def attach_trainer(self, engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.log_batch_summary)
        engine.add_event_handler(Events.STARTED, self.on_run_start)

    def attach_evaluator(self, engine, trainer, phase):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch_summary,
                                 trainer, phase)


class VisdomSummaryLogger(SummaryLogger):
    def __init__(self, run_id=None, log_interval=100, **kwargs):
        self.visdom = visdom.Visdom(**kwargs)
        self.windows = {}
        self.run_id = run_id
        self.log_interval = log_interval

        if not self.visdom.check_connection():
            raise RuntimeError(
                "Visdom server not running. Please run python -m visdom.server")

    def window_exists(self, metric):
        return metric in self.windows

    def clear_window(self, metric, trace=None):
        self.visdom.line(X=np.array([1]), Y=np.array([np.nan]),
                         update='remove',
                         name=trace,
                         win=self.windows[metric])

    def create_window(self, metric, x_label, y_label):
        if not self.window_exists(metric):
            self.windows[metric] = create_plot_window(self.visdom,
                                                      x_label=x_label,
                                                      y_label=y_label,
                                                      title=metric)

    def log_metric(self, metric, x, y, trace=None):
        self.visdom.line(X=np.array([x]),
                         Y=np.array([y]),
                         name=trace,
                         update='append',
                         win=self.windows[metric])

    def on_run_start(self, engine):
        if self.window_exists('train_loss'):
            self.clear_window('train_loss', trace='run %d' % self.run_id)
        self.run_id = self.run_id + 1 if self.run_id else 1

    def log_batch_summary(self, engine):
        metric = 'train_loss'
        if not self.window_exists(metric):
            self.create_window(metric, x_label='iteration', y_label='loss')

        iteration = engine.state.iteration - 1
        if iteration % self.log_interval == 0:
            self.log_metric(metric, x=engine.state.iteration, y=engine.state.output,
                            trace='run %d' % self.run_id)

    def log_epoch_summary(self, engine, trainer, phase):
        for name, value in engine.state.metrics.items():
            metric = '{}_{}'.format(phase, name)

            if not self.window_exists(metric):
                self.create_window(metric, x_label='epoch', y_label=name)

            self.log_metric(metric, x=trainer.state.epoch, y=value,
                            trace='run %d' % self.run_id)