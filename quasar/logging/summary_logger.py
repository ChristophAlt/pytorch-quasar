from abc import ABCMeta, abstractmethod

from ignite.engine import Events


class RunSummaryLogger(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.run_name = None

    def new_run(self, run_name):
        self.run_name = run_name

    @abstractmethod
    def on_run_start(self, trainer):
        pass

    @abstractmethod
    def batch_summary(self, evaluator):
        pass

    @abstractmethod
    def epoch_summary(self, evaluator, trainer, phase):
        pass

    @abstractmethod
    def run_summary(self, config, loss):
        pass

    def attach_trainer(self, trainer):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.batch_summary)
        trainer.add_event_handler(Events.STARTED, self.on_run_start)

    def attach_evaluator(self, evaluator, trainer, phase):
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_summary,
                                    trainer, phase)
