from __future__ import division

from sklearn import metrics

from itertools import chain

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


def flatten(y):
    """
    Flatten a list of lists.
    >>> flatten([[1,2], [3,4]])
    [1, 2, 3, 4]
    """
    return list(chain.from_iterable(y))


class F1Score(Metric):
    """
    Calculates the weighted average F1 score.
    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, labels=None, output_transform=lambda x: x):
        super(F1Score, self).__init__(output_transform)
        self.labels = labels

    def reset(self):
        self._y = []
        self._y_pred = []

    def update(self, output):
        y_pred, y = output
        self._y.extend(y.cpu().numpy().tolist())
        self._y_pred.extend(y_pred.cpu().numpy().tolist())

    def compute(self):
        if len(self._y) == 0:
            raise NotComputableError(
                'Metric must have at least one example before it can be computed')
        return metrics.f1_score(flatten(self._y), flatten(self._y_pred), average='weighted', labels=self.labels)
