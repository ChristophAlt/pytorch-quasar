import os

import torch
import dill
import subprocess

from pathlib import Path

from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import datasets_iterator, flatten_parameters

from ignite.engine import _prepare_batch

from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import datasets_iterator, flatten_parameters

from quasar.datasets import conll_dataset
from quasar.hparams.hp_optimizer import Args

from sequence_tagging import collate_fn


def load_encoder(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def load_model(path, model_filename):
    text_encoder = load_encoder(os.path.join(path, 'text_encoder.pkl'))
    character_encoder = load_encoder(os.path.join(path, 'character_encoder.pkl'))
    label_encoder = load_encoder(os.path.join(path, 'label_encoder.pkl'))
    subword_encoder = load_encoder(os.path.join(path, 'subword_encoder.pkl'))
    model = torch.load(os.path.join(path, model_filename))
    flatten_parameters(model)
    model.eval()
    
    return model, (text_encoder, character_encoder, label_encoder, subword_encoder)


def apply_model(model, example, text_encoder, label_encoder, collate_fn, gpu, use_crf):
    device = torch.device(gpu)

    model.eval()
    with torch.autograd.no_grad():
        
        (text, text_lengths, char, char_lengths, subword, subword_lengths), y = _prepare_batch(collate_fn([example]), device=device)
        
        y_pred = model.predict(text, text_lengths, char, subword,
                               char_lengths, subword_lengths, use_crf=use_crf)
        
        pred_labels = [label_encoder.itos[index] for index in y_pred.cpu().numpy().tolist()[0]]
        true_labels = [label_encoder.itos[index] for index in y.cpu().numpy().tolist()[0]]
        
        return (example['raw_text'], text_encoder.decode(example['text'])), (true_labels, pred_labels)


def call_eval_script(model, test, text_encoder, label_encoder, collate_fn, gpu, use_crf, eval_file):
    results = []

    for example in test:
        (raw_text, _), (y_true, y_pred) = apply_model(model, example, text_encoder, label_encoder,
                                                      collate_fn, gpu, use_crf)
        
        for token, y_t, y_p in zip(raw_text, y_true, y_pred):
            results.append('%s %s %s\n' % (token, y_t, y_p))
        results.append('\n')

    p = subprocess.run([eval_file], input=''.join(results), stdout=subprocess.PIPE, encoding='utf-8')
    return p.stdout, p.returncode


def main():

    args = {
        'test_file': '/home/christoph/Downloads/test.conll',
        'model_path': '/tmp/models',
        'gpu': 0,
        'batch_size': 64,
        'use_crf': True,
        'eval_file': '/home/christoph/research/pytorch-crf-tagger/eval/eval' 
    }

    args = Args(**args)

    model_path = Path(args.model_path)

    model_filename = list(model_path.glob('checkpoint_model*.pth'))[0]

    model, (text_encoder, character_encoder, label_encoder, subword_encoder) = load_model(model_path, model_filename)

    test = conll_dataset(directory='',
                         train=False,
                         dev=False,
                         test=True,
                         test_filename=args.test_file,
                         column_names=['raw_text', 'd1', 'd2', 'raw_label'],
                         use_cols=[0, 3],
                         tag_scheme='iob')

    # Encode dataset
    for ex in datasets_iterator(test):
        ex['char'] = [character_encoder.encode(token.strip()) for token in ex['raw_text']]
        ex['subword'] = [subword_encoder.encode(token.strip()) for token in ex['raw_text']]
        ex['text'] = text_encoder.encode([t.strip() for t in ex['raw_text']])
        ex['label'] = label_encoder.encode(ex['raw_label'])

    output, output_err = call_eval_script(model, test, text_encoder, label_encoder, collate_fn,
                                          args.gpu, args.use_crf, args.eval_file)

    print ('Eval script results:')
    print(output)

if __name__ == '__main__':
    main()