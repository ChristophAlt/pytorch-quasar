import torch


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.size(0)

    if max_len is None:
        max_len = lengths.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lengths.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lengths.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask
