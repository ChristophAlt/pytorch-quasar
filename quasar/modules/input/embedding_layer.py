import collections

import torch

from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_modules):
        super(EmbeddingLayer, self).__init__()
        self.embedding_modules = embedding_modules
        [self.add_module(n, m) for n, m in self.embedding_modules.items()]

    def forward(self, inputs):
        embedded_inputs = []
        for name, input_ in inputs.items():
            if isinstance(input_, collections.Sequence):
                tensor, length = input_
                embedding = self.embedding_modules[name](tensor, length)
            else:
                embedding = self.embedding_modules[name](input_)
            embedded_inputs.append(embedding)

        return torch.cat(embedded_inputs, dim=-1)

    @property
    def embedding_dim(self):
        return sum([m.embedding_dim for m in self.embedding_modules.values()])
