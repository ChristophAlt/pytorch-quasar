import os
import re
import codecs

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_files_maybe_extract


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """ Check if tags have valid IOB format.
    If tags are in IOB1 format, they are converted to IOB2.

    Args:
        tags (list): list of tags

    Returns:
        bool: true if valid IOB format, false otherwise

    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """Transform tags in IOB format to IOBES format

    Args:
        tags (list): tags in IOB format

    Returns:
        list: tags in IOBES format
    """

    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def load_sentences(path):
    """Load sentences from ConLL formated file.
    A line contains a word followed by one or multiple tags.
    Sentences are separated by empty lines.

    Args:
        path (str): path to the ConLL file

    Returns:
        list: sentences extracted from the given file. Each sentence consists
        of a list [word, tag_1, ..., tag_n]
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme.lower() == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme.lower() == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def conll_dataset(directory='data/',
                  train=False,
                  dev=False,
                  test=False,
                  train_filename='train.txt',
                  dev_filename='dev.txt',
                  test_filename='test.txt',
                  check_files=None,
                  urls=None,
                  tag_scheme=None,
                  column_names=None,
                  use_cols=None):
    """
    Load a dataset in ConLL format.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the dev split.
        test_filename (str, optional): The filename of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        urls (str, optional): URLs to download.
        tag_scheme (str, optional): The tag scheme of the contained tags (IOB or IOBES).
        column_names (str, optional): The names of the columns contained in the dataset (defaults to ConLL2003 [text, pos, chunk, entity]).
        use_cols (int, optional): The columns to retain in the dataset (defaults to all).

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training tokens, dev
        tokens and test tokens in order if their respective boolean argument is true.
    """

    urls = urls or []
    check_files = check_files or []

    download_files_maybe_extract(urls=urls, directory=directory, check_files=check_files)

    if tag_scheme and tag_scheme.lower() not in ['iob', 'iobes']:
        raise ValueError("Unknown tag scheme '%s'" % tag_scheme)

    column_names = column_names or ['text', 'pos', 'chunk', 'entity']
    use_cols = use_cols or list(range(len(column_names)))

    ret = []
    splits = [(train, train_filename), (dev, dev_filename),
              (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]

    for filename in splits:
        full_path = os.path.join(directory, filename)
        examples = []

        sentences = load_sentences(full_path)

        if tag_scheme:
            update_tag_scheme(sentences, tag_scheme)

        for sentence in sentences:
            columns = list(zip(*sentence))
            examples.append({column_names[col_idx]: list(columns[col_idx])
                             for col_idx in use_cols})

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def conll2003_dataset(directory='data/',
                      train=False,
                      dev=False,
                      test=False,
                      train_filename='eng.train',
                      dev_filename='eng.testa',
                      test_filename='eng.testb',
                      check_files=['eng.train', 'eng.testa', 'eng.testb'],
                      urls=[
                          'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train'
                          'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa',
                          'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb'
                      ],
                      tag_scheme=None,
                      column_names=None,
                      use_cols=None):
    """
    Load the ConLL2003 NER dataset.

    References:
        * https://www.clips.uantwerpen.be/conll/ner/

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the dev split.
        test_filename (str, optional): The filename of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        urls (str, optional): URLs to download.
        tag_scheme (str, optional): The tag scheme of the contained tags (IOB or IOBES).
        column_names (str, optional): The names of the columns contained in the dataset (defaults to ConLL2003 [text, pos, chunk, entity]).
        use_cols (int, optional): The columns to retain in the dataset (defaults to all).
        lower (bool, optional): If to lowercase the text column.
        zero_digits (bool, optional): If to replace digits in the text column with zeros.


    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training tokens, dev
        tokens and test tokens in order if their respective boolean argument is true.


    Example:
        >>> from quasar.datasets import conll2003_dataset
        >>> train = conll2003_dataset(train=True)
        >>> train[0]
        {
            'text': ['CRICKET', '-', 'LEICESTERSHIRE', 'TAKE', 'OVER', 'AT', 'TOP'],
            'pos': ['NNP', ':', 'NNP', 'NNP', 'IN', 'NNP', 'NNP'],
            'chunk': ['I-NP', 'O', 'I-NP', 'I-NP', 'I-PP', 'I-NP', 'I-NP'],
            'entity': ['O', 'O', 'I-ORG', 'O', 'O', 'O', 'O']
        }
    """

    return conll_dataset(directory, train, dev, test, train_filename, dev_filename,
                         test_filename, check_files, urls, tag_scheme, column_names,
                         use_cols)
