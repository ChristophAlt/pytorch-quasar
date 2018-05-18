import mock

from quasar.datasets import conll_dataset, conll2003_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/conll'


@mock.patch("urllib.request.urlretrieve")
def test_conll2003_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    train = conll2003_dataset(directory=directory, train=True, check_files=['eng.train'])

    assert len(train) > 0
    assert train[0:2] == [{
            'text': ['token', 'TOKEN', 'Token', 'token4321', 'token'],
            'pos': ['NNP', ':', 'NNP', 'NNP', 'IN'],
            'chunk': ['I-NP', 'O', 'I-NP', 'I-NP', 'I-PP'],
            'entity': ['O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
        }, {
            'text': ['Token', 'token1234'],
            'pos': ['NNP', 'CD'],
            'chunk': ['I-NP', 'I-NP'],
            'entity': ['I-LOC', 'O']
        }
    ]


@mock.patch("urllib.request.urlretrieve")
def test_conll2003_tag_scheme(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    train = conll2003_dataset(directory=directory, train=True,
                              check_files=['eng.train'], tag_scheme='iob')

    assert len(train) > 0
    assert train[0:2] == [{
            'text': ['token', 'TOKEN', 'Token', 'token4321', 'token'],
            'pos': ['NNP', ':', 'NNP', 'NNP', 'IN'],
            'chunk': ['I-NP', 'O', 'I-NP', 'I-NP', 'I-PP'],
            'entity': ['O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
        }, {
            'text': ['Token', 'token1234'],
            'pos': ['NNP', 'CD'],
            'chunk': ['I-NP', 'I-NP'],
            'entity': ['B-LOC', 'O']
        }
    ]

    train = conll2003_dataset(directory=directory, train=True,
                              check_files=['eng.train'], tag_scheme='iobes')

    assert len(train) > 0
    assert train[0:2] == [{
            'text': ['token', 'TOKEN', 'Token', 'token4321', 'token'],
            'pos': ['NNP', ':', 'NNP', 'NNP', 'IN'],
            'chunk': ['I-NP', 'O', 'I-NP', 'I-NP', 'I-PP'],
            'entity': ['O', 'B-ORG', 'I-ORG', 'E-ORG', 'O']
        }, {
            'text': ['Token', 'token1234'],
            'pos': ['NNP', 'CD'],
            'chunk': ['I-NP', 'I-NP'],
            'entity': ['S-LOC', 'O']
        }
    ] 
