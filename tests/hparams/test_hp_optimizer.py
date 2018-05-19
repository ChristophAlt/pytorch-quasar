from quasar.hparams import HPOptimizer, Args

from skopt.space import Real, Integer, Categorical


def test_hp_optimizer():
    static_params = Args(**{'d': 10})

    def objective_func(params):
        return params.a - params.b * params.c + params.d

    trials = []
    def test_callback(params, result):
        trials.append((params, result))
        assert len(params) == 3
        assert 0 <= params['a'] <= 1.5
        assert 3 <= params['b'] <= 6
        assert params['c'] in [-2, 4, 6]

    space = [
        Real(0, 1.5, name='a'),
        Integer(3, 6, name='b'),
        Categorical([-2, 4, 6], name='c')
    ]

    hp_opt = HPOptimizer(args=static_params,
                         strategy='gp',
                         space=space)
    
    hp_opt.add_callback(test_callback)

    result = hp_opt.minimize(objective_func, n_calls=10)

    assert len(trials) == 10
