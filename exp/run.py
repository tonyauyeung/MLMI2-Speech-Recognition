from dataloader import get_dataloader
import torch
from collections import Counter
from datetime import datetime
from trainer import train
import models
from decoder import decode
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description = 'Running MLMI2 experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_json', type=str, default="train_origin_fbank.json")
parser.add_argument('--val_json', type=str, default="dev_fbank.json")
parser.add_argument('--test_json', type=str, default="test_fbank.json")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=1, help="number of rnn layers")
parser.add_argument('--fbank_dims', type=int, default=23, help="filterbank dimension")
parser.add_argument('--model_dims', type=int, default=128, help="model size for rnn layers")
parser.add_argument('--concat', type=int, default=1, help="concatenating frames")
parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
parser.add_argument('--vocab', type=str, default="vocab_39.txt", help="vocabulary file path")
parser.add_argument('--report_interval', type=int, default=50, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--gradclip', type=float, default=0, help='0: donot clip; else: clip with max norm = gradclip, e.g. 1')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--mlp_layers', type=int, default=1)
parser.add_argument('--bidirection', type=bool, default=True)
args = parser.parse_args()

# import os
# os.chdir('exp/')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
vocab = {}
with open(args.vocab) as f:
    for id, text in enumerate(f):
        vocab[text.strip()] = id

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(args)
args.device = device
args.vocab = vocab

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
