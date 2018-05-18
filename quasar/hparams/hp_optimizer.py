import numpy as np

from skopt import gp_minimize, dummy_minimize
from skopt.utils import use_named_args


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class HPOptimizer:
    def __init__(self, args, space, strategy='gp', random_state=0):
        self.static_args = vars(args)
        self.space = space
        self.random_state = np.random.RandomState(seed=random_state)
        
        self.minimizer = gp_minimize if strategy == 'gp' else dummy_minimize
        self.callbacks = []
    
    def _invoke_callbacks(self, args, result):
        for callback in self.callbacks:
            callback(args, result)
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def minimize(self, func, n_calls):
        @use_named_args(self.space)
        def inner_func(**hp_args):
            combined_args = dict(self.static_args)
            combined_args.update(hp_args)
            result = func(Args(**combined_args))
            self._invoke_callbacks(hp_args, result)
            return result
        
        return self.minimizer(inner_func, dimensions=self.space, n_calls=n_calls)
