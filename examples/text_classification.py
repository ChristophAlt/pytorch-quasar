import os

import visdom
import torch
import torch.nn.functional as F

from pathlib import Path

from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import CategoricalAccuracy, Loss
from torch import nn
from torch.utils.data import DataLoader

from torchnlp.datasets import trec_dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import datasets_iterator, pad_batch
from torchnlp.text_encoders import WhitespaceEncoder
from torchnlp import word_to_vector

from test_tube import HyperOptArgumentParser

from distutils.util import strtobool

from quasar.nlp.encoders import LabelEncoder
from quasar.nlp.utils.data.sampler import FlexibleBucketBatchSampler
from quasar.train.train_model import train_model
from quasar.train.utils import train_test_split_sampler


class LSTMClassifier(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, embedding, num_layers=1,
                 bidirectional=True, dropout=.1):
        super(LSTMClassifier, self).__init__()

        self.embedding = embedding

        self.encoder = nn.LSTM(input_size=d_in, hidden_size=d_hidden,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True, dropout=dropout)

        self.n_directions = 2 if bidirectional else 1

        d_hidden = self.n_directions * d_hidden

        self.output_layer = nn.Sequential(*[
            nn.Linear(d_hidden, d_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, d_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, d_out, bias=False)
        ])

    def forward(self, text):
        text_emb = self.embedding(text)
        text_enc, _ = self.encoder(text_emb)
        text_repr = text_enc[:, -1, :]
        return F.log_softmax(self.output_layer(text_repr))


def collate_fn(batch):
    text_batch, _ = pad_batch([row['text'] for row in batch])
    label_batch = [row['label'] for row in batch]

    to_tensor = (
        lambda b: torch.stack(b).squeeze(-1))

    return text_batch, to_tensor(label_batch)


def main():
    ROOT_DIR = os.path.join(str(Path.home()), '.torchtext')

    # define parameters and hyperparameters
    parser = HyperOptArgumentParser(strategy='random_search')
    
    parser.add_argument('--data_dir', default=ROOT_DIR)
    parser.add_argument('--vector_cache_dir', default=os.path.join(ROOT_DIR, 'vector_cache'))
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--use_cuda', type=strtobool, default=torch.cuda.is_available())
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--dev_size', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_interval', default=100)
    parser.add_argument('--tune', type=strtobool, default=False)
    parser.add_argument('--checkpoint', type=strtobool, default=True)

    parser.add_argument('--bidirectional', type=strtobool, default=True)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--word_vectors', default='glove.840B.300d')
    parser.add_argument('--d_embedding', type=int, default=300)
    parser.add_argument('--word_vectors_freeze', type=strtobool, default=True)
    parser.add_argument('--early_stopping', type=strtobool, default=False)
    
    parser.opt_list('--batch_size', type=int, default=16, options=[8, 16, 32, 64, 128], tunable=True)
    parser.opt_list('--lr', type=float, default=1e-3, options=[1e-1, 1e-2, 1e-3, 1e-4], tunable=True)
    parser.opt_list('--momentum', type=float, default=.9, options=[.9], tunable=True)

    parser.opt_list('--d_hidden', type=int, default=128, options=[32, 64, 128, 256], tunable=True)
    parser.opt_list('--dropout', type=float, default=.1, options=[.2, .3, .4, .5], tunable=True)

    args = parser.parse_args()

    vis = visdom.Visdom()
    if not vis.check_connection():
        raise RuntimeError(
            "Visdom server not running. Please run python -m visdom.server")

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    # Load dataset splits
    train, test = trec_dataset(train=True, test=True, directory=args.data_dir)

    # Create encoders (TODO: best way to persist those?)
    text_corpus = [row['text'] for row in datasets_iterator(train, test)]
    text_encoder = WhitespaceEncoder(text_corpus)

    label_corpus = [row['label'] for row in datasets_iterator(train, test)]
    label_encoder = LabelEncoder(label_corpus)

    # encode dataset splits
    for row in datasets_iterator(train, test):
        row['text'] = text_encoder.encode(row['text'])
        row['label'] = label_encoder.encode(row['label'])

    # compute train / dev split for dataloader
    train_sampler, dev_sampler = train_test_split_sampler(train,
                                                          test_size=args.dev_size,
                                                          random_state=args.seed)

    # train function
    def train_f(config):
        train_batch_sampler = FlexibleBucketBatchSampler(
            train, config.batch_size,
            sampler=train_sampler,
            drop_last=True,
            sort_key=lambda r: len(row['text']))

        train_loader = DataLoader(
            train,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            pin_memory=config.use_cuda,
            num_workers=0)

        dev_batch_sampler = FlexibleBucketBatchSampler(
            train, config.test_batch_size,
            drop_last=True,
            sampler=dev_sampler,
            sort_key=lambda r: len(row['text']))

        dev_loader = DataLoader(
            train,
            batch_sampler=dev_batch_sampler,
            collate_fn=collate_fn,
            pin_memory=config.use_cuda,
            num_workers=0)

        test_sampler = BucketBatchSampler(
            test, config.test_batch_size,
            drop_last=True,
            sort_key=lambda r: len(row['text']))

        test_loader = DataLoader(
            test,
            batch_sampler=test_sampler,
            collate_fn=collate_fn,
            pin_memory=config.use_cuda,
            num_workers=0)

        embedding = nn.Embedding(text_encoder.vocab_size, config.d_embedding)

        if config.word_vectors_freeze:
            embedding.weight.requires_grad = False

        if config.word_vectors:
            # Load word vectors
            word_vectors = word_to_vector.aliases[config.word_vectors](cache=config.vector_cache_dir)
            for i, token in enumerate(text_encoder.vocab):
                embedding.weight.data[i] = word_vectors[token]
            print('Found vectors for %d tokens in vocabulary' %
                  len([t for t in text_encoder.vocab if t in word_vectors.stoi]))

        model = LSTMClassifier(d_in=embedding.embedding_dim,
                               d_out=label_encoder.vocab_size,
                               d_hidden=config.d_hidden,
                               dropout=config.dropout,
                               embedding=embedding)
        model.to(device)

        optimizer_params = list(
            filter(lambda p: p.requires_grad, model.parameters()))

        optimizer = torch.optim.SGD(optimizer_params, lr=config.lr,
                                    momentum=config.momentum)

        trainer = create_supervised_trainer(model, optimizer, F.nll_loss,
                                            device=device)

        evaluator_train = \
            create_supervised_evaluator(model,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                            'nll': Loss(F.nll_loss)},
                                        device=device)

        evaluator_dev = \
            create_supervised_evaluator(model,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                            'nll': Loss(F.nll_loss)},
                                        device=device)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch_: 1. / (1 + 0.05 * (epoch_ - 1)))

        # scoring function for early stopping and checkpointing
        def score_function(engine):
            dev_loss = engine.state.metrics['nll']
            return -dev_loss

        early_stopping = EarlyStopping(patience=15, score_function=score_function,
                                        trainer=trainer)

        def checkpoint_score_function(engine):
            dev_accuracy = engine.state.metrics['accuracy']
            return dev_accuracy

        checkpoint = ModelCheckpoint('/tmp/models', 'checkpoint', score_function=checkpoint_score_function,
                                     n_saved=1, create_dir=True, score_name="dev_accuracy")

        # lets train!
        train_model(model=model,
                    trainer=trainer,
                    epochs=config.epochs,
                    evaluator_train=evaluator_train,
                    evaluator_dev=evaluator_dev,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    lr_scheduler=lr_scheduler,
                    early_stopping=early_stopping if config.early_stopping else None,
                    checkpoint=checkpoint if config.checkpoint else None,
                    visdom=vis)

        # load checkpointed (best) model and evaluate on test loader
        model_path = Path('/tmp/models/')
        model = torch.load(list(model_path.glob('checkpoint_model*.pth'))[0])

        test_evaluator = \
            create_supervised_evaluator(model,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                            'nll': Loss(F.nll_loss)},
                                        device=device)

        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics
        print("Test Results: Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            metrics['accuracy'], metrics['nll']))

    # hyperparameter tuning!
    if args.tune:
        for config in args.trials(20):
            print(config)
            train_f(config)
    else:
        print(vars(args))
        train_f(args)


if __name__ == '__main__':
    main()
