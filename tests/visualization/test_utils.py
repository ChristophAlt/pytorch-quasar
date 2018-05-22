from quasar.visualization.utils import trials_to_dimensions


def test_trials_to_dimensions():
    trials = [
            {'lr': 1e-3, 'batch_size': 8, 'optim': 'sgd', 'large': 4.7859},
            {'lr': 1e-2, 'batch_size': 8, 'optim': 'adam', 'large': 3.0},
            {'lr': 1e-4, 'batch_size': 32, 'optim': 'adadelta', 'large': 42.425}
        ]

    dimensions = trials_to_dimensions(trials)

    print(dimensions)

    assert len(dimensions) == 4
    assert [d['label'] for d in dimensions] == ['lr', 'batch_size', 'optim', 'large']

    assert [d['range'] for d in dimensions] == [
        [1e-4, 1e-2],
        [8, 32],
        [0, 2],
        [3, 42.425]
    ]

    assert [d['values'] for d in dimensions] == [
        [1e-3, 1e-2, 1e-4],
        [8, 8, 32],
        [2, 1, 0],
        [4.7859, 3.0, 42.425]
    ]

    assert [d['tickvals'] for d in dimensions] == [
        [1e-4, 1e-3, 1e-2],
        [8, 32],
        [0, 1, 2],
        [3.0, 4.7859, 42.425]
    ]

    assert [d['ticktext'] for d in dimensions] == [
        ['0.0001', '0.001', '0.01'],
        ['8', '32'],
        ['adadelta', 'adam', 'sgd'],
        ['3', '4.7859', '42.425']
    ]
