import torch

import torch.nn.functional as F

from torch import nn
from torch.nn import init

from .rnn.bilstm import BiLSTM
from .utils import sequence_mask


class LSTMCRF(nn.Module):
    def __init__(self, crf, d_hidden, num_layers, dropout,
                 word_embedding=None, char_embedding=None, subword_embedding=None):
        super(LSTMCRF, self).__init__()

        self.crf = crf
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.subword_embedding = subword_embedding

        self.d_hidden = d_hidden

        self.d_embedding = \
            sum(emb.embedding_dim for emb in [word_embedding, char_embedding, subword_embedding]
                if emb is not None)

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

        #res = [emb(x) for emb, x in zip(self.embeddings, xs)]
        #x = torch.cat(res, 2)
        
        emb_xs = self.word_embedding(xs) if self.word_embedding is not None else None
        emb_xc = self.char_embedding(xc, len_xc) if self.char_embedding is not None else None
        emb_xsub = self.subword_embedding(xsub, len_xsub) if self.subword_embedding is not None else None

        x = torch.cat([e for e in (emb_xs, emb_xc, emb_xsub) if e is not None], dim=-1)
        
        return x

    def _forward_bilstm(self, xs, len_xs, xc, xsub, len_xc, len_xsub):
        batch_size, seq_len = xs.size()

        x = self._embeddings(xs, xc, xsub, len_xc, len_xsub)
        x = x.view(-1, self.d_embedding)
        x = self.relu(self.input_layer(x))
        x = x.view(batch_size, seq_len, self.d_hidden)

        x = self.dropout(x)
        #o, h = self._run_rnn_packed(self.lstm, x, lens)
        o, h = self.lstm(x, len_xs)
        o = self.dropout(o)

        o = o.contiguous()
        o = o.view(-1, self.d_hidden_out)
        o = self.relu(self.output_layer(o))
        o = o.view(batch_size, seq_len, self.n_labels)

        return o

    def _bilstm_score(self, logits, y, len_xs):
        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(len_xs).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def score(self, xs, y, xc, len_xs, xsub, len_xc, len_xsub, logits=None):
        if logits is None:
            logits = self._forward_bilstm(xs, len_xs, xc, xsub, len_xc, len_xsub)

        transition_score = self.crf.transition_score(y, len_xs)
        bilstm_score = self._bilstm_score(logits, y, len_xs)

        score = transition_score + bilstm_score

        return score

    def predict(self, xs, len_xs, xc, xsub, len_xc, len_xsub, use_crf, return_scores=False):
        logits = self._forward_bilstm(xs, len_xs, xc, xsub, len_xc, len_xsub)
        
        if use_crf:
            scores, preds = self.crf.viterbi_decode(logits, len_xs)
        else:
            scores, preds = torch.max(logits, dim=-1)

        if return_scores:
            return preds, scores
        else:
            return preds

    def neg_log_likelihood(self, xs, y, len_xs, xc, xsub, len_xc, len_xsub, use_crf, return_logits=False):
        batch_size, seq_len = xs.size()
        
        logits = self._forward_bilstm(xs, len_xs, xc, xsub, len_xc, len_xsub)
        
        if use_crf:
            norm_score = self.crf(logits, len_xs)
            sequence_score = self.score(xs, y, xc, len_xs, xsub, len_xc, len_xsub, logits=logits)
            loglik = -(sequence_score - norm_score)
        else:
            loglik = (F.cross_entropy(logits.view(-1, self.n_labels), y.view(-1), reduce=False)
                      .view(batch_size, seq_len)
                      .sum(dim=-1))

        if return_logits:
            return loglik, logits
        else:
            return loglik
