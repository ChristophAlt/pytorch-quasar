import mock

from quasar.datasets import conll_dataset, conll2003_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/conll'


@mock.patch("urllib.request.urlretrieve")
def test_conll2003_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train = conll2003_dataset(directory=directory, train=True, check_files=['eng.train'])

    assert len(train) > 0
    assert train[0:2] == [{
            'text': ['CRICKET', '-', 'LEICESTERSHIRE', 'TAKE', 'OVER', 'AT', 'TOP'],
            'pos': ['NNP', ':', 'NNP', 'NNP', 'IN', 'NNP', 'NNP'],
            'chunk': ['I-NP', 'O', 'I-NP', 'I-NP', 'I-PP', 'I-NP', 'I-NP'],
            'entity': ['O', 'O', 'I-ORG', 'O', 'O', 'O', 'O']
        }, {
            'text': ['LONDON', '1996-08-30'],
            'pos': ['NNP', 'CD'],
            'chunk': ['I-NP', 'I-NP'],
            'entity': ['I-LOC', 'O']
        }
    ]
