import torch

from torchnlp.text_encoders import PADDING_INDEX

from torchnlp.utils import datasets_iterator, pad_batch, pad_tensor


def pad_nested_batch(batch, padding_index=PADDING_INDEX):
    max_len_h = max([len(row) for row in batch])
    max_len = max([len(t) for row in batch for t in row])

    lengths = [[len(t) for t in row] + [0] * (max_len_h - len(row)) for row in batch]
    batch = [row + [torch.LongTensor(max_len).fill_(padding_index)] * (max_len_h - len(row)) for row in batch]

    padded = torch.stack([torch.stack([pad_tensor(t, max_len, padding_index) for t in row]) for row in batch]).contiguous()
    return padded, lengths
