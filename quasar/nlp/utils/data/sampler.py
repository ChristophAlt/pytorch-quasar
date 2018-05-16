from torch.utils.data.sampler import RandomSampler, BatchSampler
from torchnlp.samplers import BucketBatchSampler


class FlexibleBucketBatchSampler(BucketBatchSampler):
    def __init__(self, data, batch_size, drop_last, sampler=None, **kwargs):
        super().__init__(data, batch_size, drop_last, **kwargs)

        sampler = sampler or RandomSampler(data)

        self.bucket_sampler = BatchSampler(
            sampler, batch_size * self.bucket_size_multiplier, False)
