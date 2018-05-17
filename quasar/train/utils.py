import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def train_test_split_sampler(dataset, test_size=None, shuffle=True,
                             random_state=None):

    n_examples = len(dataset)
    indices = list(range(n_examples))

    if test_size is None:
        test_size = .2

    split = int((1 - test_size * n_examples))

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, dev_sampler
