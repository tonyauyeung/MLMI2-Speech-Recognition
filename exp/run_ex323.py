from dataloader import get_dataloader
import torch
from collections import Counter
from datetime import datetime
import models
import numpy as np
import argparse
import random
from jiwer import compute_measures, cer

from datetime import datetime
from pathlib import Path

from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax, softmax
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import concat_inputs
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser(description = 'Running MLMI2 experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_json', type=str, default="train_p61_fbank.json")
parser.add_argument('--val_json', type=str, default="dev_fbank.json")
parser.add_argument('--test_json', type=str, default="test_fbank.json")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=1, help="number of rnn layers")
parser.add_argument('--fbank_dims', type=int, default=23, help="filterbank dimension")
parser.add_argument('--model_dims', type=int, default=128, help="model size for rnn layers")
parser.add_argument('--concat', type=int, default=1, help="concatenating frames")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--vocab', type=str, default="phone_map.txt", help="vocabulary file path")
parser.add_argument('--report_interval', type=int, default=50, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--gradclip', type=float, default=1.0, help='0: donot clip; else: clip with max norm = gradclip, e.g. 1')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--mlp_layers', type=int, default=1)
parser.add_argument('--bidirection', type=bool, default=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# import os
# os.chdir('exp/')

vocab = {}
phone_map = {}
with open(args.vocab) as f:
    vocab['_'] = 0
    phone_map['_'] = '_'
    for idx, text in enumerate(f):
        key, value = text.strip().split(':')
        phone_map[key] = value[1:]
        vocab[key] = idx + 1
vocab_mapped = {}
# with open('vocab_39.txt') as f:
#     for id, text in enumerate(f):
#         vocab_mapped[text.strip()] = idx
# idx_map = {}
# for key, value in vocab.items():
#     idx_map[value] = vocab_mapped[phone_map[key]]

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(args)
args.device = device
args.vocab = vocab

# def origin2map(origin_probs):
#     mapped_probs = torch.zeros((origin_probs.shape[0], origin_probs.shape[1], 40), device=args.device)
#     for key, value in idx_map.items():
#         mapped_probs[:, :, value] += origin_probs[:, :, key]
#     return mapped_probs

def decode(model, args, json_file, char=False):
    idx2grapheme = {y: x for x, y in vocab.items()}
    test_loader = get_dataloader(json_file, 1, False)
    stats = [0., 0., 0., 0.]
    for data in test_loader:
        inputs, in_lens, trans, _ = data
        inputs = inputs.to(args.device)
        in_lens = in_lens.to(args.device)
        inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(inputs), dim=-1)
            outputs = torch.argmax(outputs, dim=-1).transpose(0, 1)
        outputs = [[phone_map[idx2grapheme[i]] for i in j] for j in outputs.tolist()]
        outputs = [[v for i, v in enumerate(j) if i == 0 or v != j[i - 1]] for j in outputs]
        outputs = [list(filter(lambda elem: elem != "_", i)) for i in outputs]
        outputs = [" ".join(i) for i in outputs]
        if char:
            cur_stats = cer(trans, outputs, return_dict=True)
        else:
            cur_stats = compute_measures(trans, outputs)
        stats[0] += cur_stats["substitutions"]
        stats[1] += cur_stats["deletions"]
        stats[2] += cur_stats["insertions"]
        stats[3] += cur_stats["hits"]

    total_words = stats[0] + stats[1] + stats[3]
    sub = stats[0] / total_words * 100
    dele = stats[1] / total_words * 100
    ins = stats[2] / total_words * 100
    cor = stats[3] / total_words * 100
    err = (stats[0] + stats[1] + stats[2]) / total_words * 100
    return sub, dele, ins, cor, err

def train(model, args):
    torch.manual_seed(args.seed)
    train_loader = get_dataloader(args.train_json, args.batch_size, True)
    val_loader = get_dataloader(args.val_json, args.batch_size, False)
    criterion = CTCLoss(zero_infinity=True)
    optimiser = Adam(model.parameters(), lr=args.lr)
    val_per = [torch.inf]
    lr = args.lr

    def train_one_epoch(epoch):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(train_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            # Gradient Clip with max norm "gradclip"
            if args.gradclip != 0:
                clip_grad_norm_(model.parameters(), args.gradclip)
            optimiser.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path('checkpoints/{}'.format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_val_per = 1e+6

    for epoch in range(args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_val_loss = 0.
        val_decode = decode(model, args, args.val_json)
        print('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%'.format(
            avg_train_loss, 0, val_decode[4]))

        # using checkpoints to perform early-stopping
        if val_decode[4] < best_val_per:
            model_path = 'checkpoints/{}/model_{}'.format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
    return model_path


model = models.BiLSTM(args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab), args.dropout, args.mlp_layers, args.bidirection)

if torch.__version__ == "2.1.0":
    model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters())
print('Total number of model parameters is {}'.format(num_params))


start = datetime.now()
model.to(args.device)
model_path = train(model, args)
end = datetime.now()
duration = (end - start).total_seconds()
print('Training finished in {} minutes.'.format(divmod(duration, 60)[0]))
print('Model saved to {}'.format(model_path))

print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)
results = decode(model, args, args.test_json)
print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%".format(*results))
