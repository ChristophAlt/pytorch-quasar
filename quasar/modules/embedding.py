import torch
import math

from torch import nn


class Embedding(nn.Module):
    def __init__(self, d_vocab=None, d_emb=None, vectors=None,
                 freeze_embedding=False, dropout=0., **kwargs):
        super(Embedding, self).__init__()

        if vectors is not None:
            d_vocab, d_emb = vectors.size()

        if not d_vocab or not d_emb:
            raise Exception("Either a tensor with embeddings or 'd_vocab' "
                            "and 'd_emb' must be provided.")

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(d_vocab, d_emb, **kwargs)

        if vectors is not None:
            self.embedding.weight = \
                nn.Parameter(vectors, requires_grad=not freeze_embedding)

    def forward(self, input):
        return self.embedding(input)

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    
class CharEmbedding(nn.Module):
    def __init__(self, d_vocab=None, d_emb=None, vectors=None,
                 freeze_embedding=False, dropout=0., **kwargs):
        super(CharEmbedding, self).__init__()

        if vectors is not None:
            d_vocab, d_emb = vectors.size()

        if not d_vocab or not d_emb:
            raise Exception("Either a tensor with embeddings or 'd_vocab' "
                            "and 'd_emb' must be provided.")

        self.embedding = nn.Embedding(d_vocab, d_emb, **kwargs)

        if vectors is not None:
            self.embedding.weight = \
                nn.Parameter(vectors, requires_grad=not freeze_embedding)
        else:
            # init vectors
            nn.init.uniform(self.embedding.weight.data,
                            -math.sqrt(3. / d_emb), math.sqrt(3. / d_emb))
            #nn.init.xavier_uniform(self.embedding.weight.data)
        
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = BiLSTM(input_size=self.d_hidden, size=d_hidden)

    def forward(self, input, lengths):
        batch_size, n_words, n_chars = input.size()
        
        # input => B x n_words x n_chars
        # => (B x n_words) x n_chars
        input = input.view(batch_size * n_words, -1)
        embed = self.embedding(input)
        # => (B x n_words) x n_chars x d_emb
        enc, _ = self.lstm(embed, lengths)
        # => (B x n_words) x n_chars x d_emb
        # => (B x n_words) x d_emb => B x n_words x d_emb
        return enc[:, -1, :].view(batch_size, n_words, -1)
        

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim * 2
    
    
class ConvCharEmbedding(nn.Module):
    def __init__(self, d_vocab=None, d_emb=None, vectors=None,
                 freeze_embedding=False, conv_width=5, **kwargs):
        super(ConvCharEmbedding, self).__init__()

        if vectors is not None:
            d_vocab, d_emb = vectors.size()

        if not d_vocab or not d_emb:
            raise Exception("Either a tensor with embeddings or 'd_vocab' "
                            "and 'd_emb' must be provided.")

        self.embedding = nn.Embedding(d_vocab, d_emb, **kwargs)
        
        if vectors is not None:
            self.embedding.weight = \
                nn.Parameter(vectors, requires_grad=not freeze_embedding)
        else:
            # init vectors
            #nn.init.uniform(self.embedding.weight.data, -sqrt(3. / d_emb), sqrt(3. / d_emb))
            self.embedding.weight.data.mul_(0.1)
            
        self.conv = torch.nn.Conv1d(d_emb, d_emb, conv_width,
                                    padding=math.floor(conv_width / 2))

    def forward(self, input, lengths):
        batch_size, n_words, n_chars = input.size()

        # input => B x n_words x n_chars
        # => (B x n_words) x n_chars
        input = input.view(batch_size * n_words, -1)
        embed = self.embedding(input)
        # => (B x n_words) x n_chars x d_emb
        conv_out = self.conv(embed.transpose(1, 2))
        
        return conv_out.max(dim=-1)[0].view(batch_size, n_words, -1)

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim
