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
import json

parser = argparse.ArgumentParser(description = 'Running MLMI2 experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_json', type=str, default="train_fbank.json")
parser.add_argument('--val_json', type=str, default="dev_fbank.json")
parser.add_argument('--test_json', type=str, default="test_fbank.json")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2, help="number of rnn layers")
parser.add_argument('--fbank_dims', type=int, default=23, help="filterbank dimension")
parser.add_argument('--model_dims', type=int, default=128, help="model size for rnn layers")
parser.add_argument('--concat', type=int, default=1, help="concatenating frames")
parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
parser.add_argument('--vocab', type=str, default="vocab_39.txt", help="vocabulary file path")
parser.add_argument('--report_interval', type=int, default=50, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--gradclip', type=float, default=0, help='0: donot clip; else: clip with max norm = gradclip, e.g. 1')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--mlp_layers', type=int, default=1)
parser.add_argument('--bidirection', type=bool, default=True)
args = parser.parse_args()

# seeds = [1, 520, 1314, 2000]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
lrs = [0.005, 0.001]
dropouts = [0.1, 0.3]
gradclips = [1.0, 5.0]
mlps = [1, 2]

vocab = {}
with open(args.vocab) as f:
    for id, text in enumerate(f):
        vocab[text.strip()] = id

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
args.device = device
args.vocab = vocab

res = {}
for lr in lrs:
    for dropout in dropouts:
        for gradclip in gradclips:
            for mlp in mlps:
                name = 'lr={}.dropout={}.gradclip={}.mlp_layers={}'.format(lr, dropout, gradclip, mlp)
                print('training: {}'.format(name))
                    # dicdic = {}
                    # setattr(args, 'seed', seed)
                setattr(args, 'lr', lr)
                setattr(args, 'dropout', dropout)
                setattr(args, 'gradclip', gradclip)
                setattr(args, 'mlp_layers', mlp)
                    # torch.manual_seed(args.seed)
                    # np.random.seed(args.seed)
                    # random.seed(args.seed)

                model = models.BiLSTM(args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab), args.dropout, args.mlp_layers, args.bidirection)

                if torch.__version__ == "2.1.0":
                    model = torch.compile(model)

                num_params = sum(p.numel() for p in model.parameters())


                start = datetime.now()
                model.to(args.device)
                model_path = train(model, args)
                end = datetime.now()
                duration = (end - start).total_seconds()

                model.load_state_dict(torch.load(model_path))
                model.eval()
                model.to(device)
                results = decode(model, args, args.test_json)
                print('Total number of model parameters is {}'.format(num_params))
                print('Training finished in {} minutes.'.format(divmod(duration, 60)[0]))
                print('Model saved to {}'.format(model_path))
                print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%\n".format(*results))
                    # dicdic[seed] = {'model_path': model_path, 'results': results, 'num_params': num_params, 'training_time': duration}
                dicdic = {'model_path': model_path, 'results': results, 'num_params': num_params, 'training_time': duration}
                res[name] = dicdic

with open('res.json', 'w') as f:
    json.dump(res, f, indent=4)