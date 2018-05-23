import os

import torch
import torch.nn.functional as F

import visdom
import dill

from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, _prepare_batch
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import CategoricalAccuracy, Loss

from torchnlp.datasets import trec_dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import datasets_iterator, pad_batch, pad_tensor
from torchnlp import word_to_vector
from torchnlp.text_encoders import IdentityEncoder, CharacterEncoder, SubwordEncoder
from torchnlp.text_encoders import PADDING_INDEX

from skopt.space import Real, Categorical

from quasar.datasets import conll_dataset
from quasar.modules.crf import CRF
from quasar.modules.lstm_crf import LSTMCRF
from quasar.nlp.encoders import LabelEncoder
from quasar.modules.embedding import Embedding, ConvCharEmbedding
from quasar.logging import VisdomSummaryLogger
from quasar.train.train_model import train_model
from quasar.train.utils import train_test_split_sampler
from quasar.hparams.hp_optimizer import HPOptimizer, Args
from quasar.visualization.utils import trials_to_dimensions
from quasar.visualization.visdom import parallel_coordinates_window
from quasar.train.metrics.f1_score import F1Score


def pad_nested_batch(batch, padding_index=PADDING_INDEX):
    max_len_h = max([len(row) for row in batch])
    max_len = max([len(t) for row in batch for t in row])

    lengths = [[len(t) for t in row] + [0] * (max_len_h - len(row)) for row in batch]
    batch = [row + [torch.LongTensor(max_len).fill_(padding_index)] * (max_len_h - len(row)) for row in batch]

    padded = [[pad_tensor(t, max_len, padding_index) for t in row] for row in batch]
    return padded, lengths


def collate_fn(batch, train=True):
    text_batch, text_lengths = pad_batch([ex['text'] for ex in batch])
    char_batch, char_lengths = pad_nested_batch([ex['char'] for ex in batch])
    subword_batch, subword_lengths = pad_nested_batch([ex['subword'] for ex in batch])
    label_batch, _ = pad_batch([ex['label'] for ex in batch])

    # stack batches into single 2d / 3d input tensors
    to_tensor = (lambda b: torch.stack(b).contiguous())
    
    to_tensor_nested = (
        lambda b: torch.stack([torch.stack(x) for x in b]).contiguous())
    
    return (text_batch, torch.LongTensor(text_lengths),
            to_tensor_nested(char_batch), torch.LongTensor(char_lengths),
            to_tensor_nested(subword_batch), torch.LongTensor(subword_lengths)), label_batch


def create_supervised_sequence_trainer(model, optimizer, parameters, device=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        (text, text_lengths, char, char_lengths, 
         subword, subword_lengths), label = _prepare_batch(batch, device=device)
        nll = model.neg_log_likelihood(text, label, text_lengths, char, subword, 
                                       char_lengths, subword_lengths, use_crf=True)
        loss = nll.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        return (nll / text_lengths.float()).mean().item()

    return Engine(_update)


def create_supervised_sequence_evaluator(model, metrics={}, device=None):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            (text, text_lengths, char, char_lengths, 
             subword, subword_lengths), label = _prepare_batch(batch, device=device)
            y_pred = model.predict(text, text_lengths, char, subword,
                                   char_lengths, subword_lengths, use_crf=True)
            nll = model.neg_log_likelihood(text, label, text_lengths, char, subword, 
                                           char_lengths, subword_lengths, use_crf=True)
            return (nll / text_lengths.float()).mean(), y_pred, label

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def save_encoder(encoder, path):
    with open(path, 'wb') as f:
        dill.dump(encoder, f)


def main():
    ROOT_DIR = os.path.join(str(Path.home()), '.torchtext')

    # define parameters and hyperparameters
    args = {
        'data_dir': ROOT_DIR,

        'train_file': '/home/christoph/Downloads/train.conll',
        'validation_file': '/home/christoph/Downloads/dev.conll',

        'use_cuda': True,
        'test_batch_size': 128,
        'dev_size': 0.1,
        'checkpoint': True,
        'early_stopping': False,
        'epochs': 35,
        'hp_tune': True,

        'dropout': 0.5,
        'batch_size': 16,
        'd_hidden': 100,
        'lr': 0.015,
        'lr_decay': 0.05,

        'use_word_embedding': False,
        'word_embedding_pretrained': 'german.model',
        'word_embedding_freeze': True,
        'word_embedding_dim': 300,
        'vector_cache_dir': os.path.join(ROOT_DIR, 'vector_cache'),

        'use_char_embedding': True,
        'char_embedding_pretrained': None,
        'char_embedding_freeze': False,
        'char_embedding_conv_width': 3,
        'char_embedding_dim': 30,

        'use_subword_embedding': True,
        'subword_embedding_pretrained': None,
        'subword_embedding_freeze': False,
        'subword_embedding_conv_width': 3,
        'subword_embedding_dim': 50,

        'use_crf': True,

        'momentum': .9,

        'seed': 42
    }

    args = Args(**args)

    vis = visdom.Visdom()
    if not vis.check_connection():
        raise RuntimeError(
            "Visdom server not running. Please run python -m visdom.server")

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    train, dev = conll_dataset(directory='',
                               train=True,
                               dev=True,
                               train_filename=args.train_file,
                               dev_filename=args.validation_file,
                               column_names=['text', 'd1', 'd2', 'label'],
                               use_cols=[0, 3],
                               tag_scheme='iob')

    text_corpus = [ex['text'] for ex in datasets_iterator(train, dev)]
    text_encoder = IdentityEncoder(text_corpus)

    subword_corpus = [token for ex in datasets_iterator(train, dev) for token in ex['text']]
    subword_encoder = SubwordEncoder(subword_corpus, target_vocab_size=750)
    
    character_corpus = [token for ex in datasets_iterator(train, dev) for token in ex['text']]
    character_encoder = CharacterEncoder(character_corpus)

    label_corpus = [ex['label'] for ex in datasets_iterator(train, dev)]
    label_encoder = LabelEncoder(['O'] + label_corpus)

    # Encode dataset
    for ex in datasets_iterator(train, dev):
        ex['char'] = [character_encoder.encode(token.strip()) for token in ex['text']]
        ex['subword'] = [subword_encoder.encode(token.strip()) for token in ex['text']]
        ex['text'] = text_encoder.encode([t.strip() for t in ex['text']])
        ex['label'] = label_encoder.encode(ex['label'])

    print('Num examples train:', len(train))
    print('Num examples dev:', len(dev))

    print('Text vocabulary size:', text_encoder.vocab_size)
    print('Character vocabulary size:', character_encoder.vocab_size)
    print('Subword vocabulary size:', subword_encoder.vocab_size)
    print('Entity vocabulary size:', label_encoder.vocab_size)
    print('Entity vocabulary:', label_encoder.vocab)

    def delete_checkpoint(path):
        checkpoint_files = list(path.glob('checkpoint_model*.pth'))
        if checkpoint_files:
            os.remove(checkpoint_files[0])

    visdom_logger = VisdomSummaryLogger()

    f1_labels = list(range(1, label_encoder.vocab_size))

    model_path = Path('/tmp/models/')

    model_path.mkdir(exist_ok=True)

    save_encoder(text_encoder, os.path.join(model_path, 'text_encoder.pkl'))
    save_encoder(character_encoder, os.path.join(model_path, 'character_encoder.pkl'))
    save_encoder(label_encoder, os.path.join(model_path, 'label_encoder.pkl'))
    save_encoder(subword_encoder, os.path.join(model_path, 'subword_encoder.pkl'))

    def train_f(config):

        delete_checkpoint(model_path)

        train_batch_sampler = BucketBatchSampler(
            train, config.batch_size,
            drop_last=True,
            sort_key=lambda row: len(row['text']))

        train_loader = DataLoader(
            train,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            pin_memory=config.use_cuda,
            num_workers=0)

        dev_batch_sampler = BucketBatchSampler(
            dev, config.test_batch_size,
            drop_last=True,
            sort_key=lambda row: len(row['text']))

        dev_loader = DataLoader(
            dev,
            batch_sampler=dev_batch_sampler,
            collate_fn=collate_fn,
            pin_memory=config.use_cuda,
            num_workers=0)

        if config.use_char_embedding:
            char_embedding = ConvCharEmbedding(d_vocab=character_encoder.vocab_size,
                                               d_emb=config.char_embedding_dim,
                                               conv_width=config.char_embedding_conv_width)
        else:
            char_embedding = None

        if config.use_word_embedding:
            word_embedding = Embedding(d_vocab=text_encoder.vocab_size,
                                       d_emb=config.word_embedding_dim,
                                       freeze_embedding=config.word_embedding_freeze,
                                       dropout=config.dropout)
            if config.word_embedding_pretrained:
                word_embedding_name = config.word_embedding_pretrained

                # if name refers to a location in the vector cache, load it
                if os.path.isfile(os.path.join(config.vector_cache_dir, word_embedding_name)):
                    word_vectors = _PretrainedEmbeddings(word_embedding_name, cache=config.vector_cache_dir)
                # if name is an alias, load it
                elif word_embedding_name in word_to_vector.aliases:
                    word_vectors = word_to_vector.aliases[word_embedding_name](cache=config.vector_cache_dir)
                else:
                    raise ValueError("Word embedding '%s' is neither an existing file nor an alias" % 
                        word_embedding_name)
                
                print('Found vectors for %d tokens in vocabulary' % 
                    len([t for t in text_encoder.vocab if t in word_vectors.stoi]))
                for i, token in enumerate(text_encoder.vocab):
                    word_embedding.embedding.weight.data[i] = word_vectors[token]
            
        else:
            word_embedding = None
        
        if config.use_subword_embedding:
            subword_embedding = ConvCharEmbedding(d_vocab=subword_encoder.vocab_size,
                                                  d_emb=config.subword_embedding_dim,
                                                  conv_width=config.subword_embedding_conv_width)
        else:
            subword_embedding = None

        crf = CRF(vocab_size=label_encoder.vocab_size)

        model = LSTMCRF(crf=crf, d_hidden=config.d_hidden, num_layers=1,
                        dropout=config.dropout,
                        word_embedding=word_embedding,
                        char_embedding=char_embedding,
                        subword_embedding=subword_embedding)

        model.to(device=device)

        optimizer_params = list(
            filter(lambda p: p.requires_grad, model.parameters()))

        optimizer = torch.optim.SGD(optimizer_params, lr=config.lr,
                                    momentum=config.momentum)

        trainer = create_supervised_sequence_trainer(model, optimizer,
                                                     parameters=optimizer_params,
                                                     device=device)

        evaluator_train = \
            create_supervised_sequence_evaluator(model,
                                        metrics={
                                            'nll': Loss(lambda x, y: x, output_transform=lambda x: (x[0], x[2])),
                                            'f1': F1Score(labels=f1_labels, output_transform=lambda x: (x[1], x[2]))
                                            },
                                        device=device)

        evaluator_dev = \
            create_supervised_sequence_evaluator(model,
                                        metrics={
                                            'nll': Loss(lambda x, y: x, output_transform=lambda x: (x[0], x[1])),
                                            'f1': F1Score(labels=f1_labels, output_transform=lambda x: (x[1], x[2]))
                                            },
                                        device=device)

        visdom_logger.attach_trainer(trainer)
        visdom_logger.attach_evaluator(evaluator_train, trainer, phase='train')
        visdom_logger.attach_evaluator(evaluator_dev, trainer, phase='dev')

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch_: 1. / (1 + config.lr_decay * (epoch_ - 1)))

        # scoring function for early stopping and checkpointing
        def score_function(engine):
            dev_loss = engine.state.metrics['nll']
            return -dev_loss

        early_stopping = EarlyStopping(patience=15, score_function=score_function,
                                       trainer=trainer)

        def checkpoint_score_function(engine):
            dev_f1 = engine.state.metrics['f1']
            return dev_f1

        checkpoint = ModelCheckpoint('/tmp/models', 'checkpoint',
                                     score_function=checkpoint_score_function,
                                     n_saved=1, create_dir=True,
                                     score_name="dev_f1")

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
                    checkpoint=checkpoint if config.checkpoint else None)

        # load checkpointed (best) model and evaluate dev set again
        model = torch.load(list(model_path.glob('checkpoint_model*.pth'))[0])

        test_evaluator = \
            create_supervised_sequence_evaluator(model,
                                        metrics={
                                            'nll': Loss(lambda x, y: x, output_transform=lambda x: (x[0], x[1])),
                                            'f1': F1Score(labels=f1_labels, output_transform=lambda x: (x[1], x[2]))
                                            },
                                        device=device)

        test_evaluator.run(dev_loader)
        metrics = test_evaluator.state.metrics
        return metrics['nll']

    if args.hp_tune:
        # hyperparameter tuning!
        hp_opt = HPOptimizer(args=args,
                            strategy='gp',
                            space=[
                                    Real(0.1, 0.5, name='dropout'),
                                    Categorical([50, 100, 150, 200], name='d_hidden'),
                                    Real(1e-4, 1, prior='log-uniform', name='lr'),
                                    Real(1e-3, 1, prior='log-uniform', name='lr_decay'),
                                    Categorical([4, 8, 10, 16], name='batch_size')
                                ])

        trials = []

        def log_trials(params, loss):
            print('Trial: ', params, 'Loss:', loss)
            params_copy = dict(params)
            params_copy['loss'] = loss
            trials.append(params_copy)

        hp_opt.add_callback(log_trials)

        result = hp_opt.minimize(train_f, n_calls=20)

        parallel_coordinates_window(vis, dimensions=trials_to_dimensions(trials),
                                    title='Hyperparameters')
    else:
        train_f(args)


if __name__ == '__main__':
    main()
