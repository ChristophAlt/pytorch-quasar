import torch

import torch.nn.functional as F

from torch import nn
from torch.nn import init

from .rnn.bilstm import BiLSTM
from .utils import sequence_mask


class LSTMCRF(nn.Module):
    def __init__(self, crf, d_hidden, num_layers, dropout,
                 embedding_layer):
        super(LSTMCRF, self).__init__()

        self.crf = crf
        self.embedding_layer = embedding_layer

        self.d_hidden = d_hidden

        self.d_embedding = self.embedding_layer.embedding_dim

        self.n_labels = self.crf.n_labels

        self.d_hidden_out = d_hidden
        #if bidirectional:
        self.d_hidden_out *= 2

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_layer = nn.Linear(self.d_embedding, self.d_hidden)
        self.output_layer = nn.Linear(self.d_hidden_out, self.n_labels)
        
        self.lstm = BiLSTM(input_size=self.d_hidden, size=d_hidden)

    def reset_parameters(self):
        #init.xavier_normal(self.word_embedding.embedding.weight.data)

        #init.xavier_normal(self.input_layer.weight.data)
        #init.xavier_normal(self.output_layer.weight.data)
        
        #init.xavier_uniform(self.input_layer.weight.data)
        init.xavier_uniform(self.output_layer.weight.data)
        
        self.crf.reset_parameters()
        self.lstm.reset_parameters()

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_lens.data.tolist(),
                                                     batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, h

    def _embeddings(self, xs, xc, xsub, len_xc, len_xsub):
        """Takes raw feature sequences and produces a single word embedding
        Arguments:
            xs: [n_feats, batch_size, seq_len] LongTensor
        Returns:
            [batch_size, seq_len, word_dim] FloatTensor
        """
        emb_xs = self.word_embedding(xs) if self.word_embedding is not None else None
        emb_xc = self.char_embedding(xc, len_xc) if self.char_embedding is not None else None
        emb_xsub = self.subword_embedding(xsub, len_xsub) if self.subword_embedding is not None else None

        x = torch.cat([e for e in (emb_xs, emb_xc, emb_xsub) if e is not None], dim=-1)
        
        return x

    def _forward_bilstm(self, input_, lengths):
        x = self.embedding_layer(input_)
        x = self.relu(self.input_layer(x))

        x = self.dropout(x)
        o, h = self.lstm(x, lengths)
        o = self.dropout(o)

        o = o.contiguous()
        o = self.relu(self.output_layer(o))

        return o

    def _bilstm_score(self, logits, y, lengths):
        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(lengths).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def score(self, input_, y, logits=None):
        inputs, lengths = input_

        if logits is None:
            logits = self._forward_bilstm(inputs, lengths)

        transition_score = self.crf.transition_score(y, lengths)
        bilstm_score = self._bilstm_score(logits, y, lengths)

        score = transition_score + bilstm_score

        return score

    def predict(self, input_, use_crf, return_scores=False):
        inputs, lengths = input_

        logits = self._forward_bilstm(inputs, lengths)
        
        if use_crf:
            scores, preds = self.crf.viterbi_decode(logits, lengths)
        else:
            scores, preds = torch.max(logits, dim=-1)

        if return_scores:
            return preds, scores
        else:
            return preds

    def neg_log_likelihood(self, input_, y, use_crf, return_logits=False):
        inputs, lengths = input_
        
        logits = self._forward_bilstm(inputs, lengths)

        batch_size, seq_len, _ = logits.size()
        
        if use_crf:
            norm_score = self.crf(logits, lengths)
            sequence_score = self.score(input_, y, logits=logits)
            loglik = -(sequence_score - norm_score)
        else:
            loglik = (F.cross_entropy(logits.view(-1, self.n_labels), y.view(-1), reduce=False)
                      .view(batch_size, seq_len)
                      .sum(dim=-1))

        if return_logits:
            return loglik, logits
        else:
            return loglik
