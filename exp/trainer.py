from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from decoder import decode
from utils import concat_inputs
from torch.nn.utils import clip_grad_norm_

from dataloader import get_dataloader

def train(model, args):
    torch.manual_seed(args.seed)
    train_loader = get_dataloader(args.train_json, args.batch_size, True)
    val_loader = get_dataloader(args.val_json, args.batch_size, False)
    criterion = CTCLoss(zero_infinity=True)
    optimiser = Adam(model.parameters(), lr=args.lr) if args.optimizer == 'Adam' else SGD(model.parameters(), lr=args.lr, momentum=0.9)
    val_losses = [torch.inf]
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
    best_val_loss = 1e+6

    for epoch in range(args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_val_loss = 0.
        for idx, data in enumerate(val_loader):
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
            outputs = log_softmax(model(inputs), dim=-1)
            val_loss = criterion(outputs, targets, in_lens, out_lens)
            running_val_loss += val_loss
        avg_val_loss = running_val_loss / len(val_loader)
        s = ''
        if args.optimizer == 'scheduled_SGD':
            if avg_val_loss >= val_losses[-1]:
                lr *= 0.5
                for g in optimiser.param_groups:
                    g['lr'] = lr
                s = '\tlr changes to {:.5f}'.format(lr)
            val_losses.append(avg_val_loss)
        val_decode = decode(model, args, args.val_json)
        print('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%'.format(
            avg_train_loss, avg_val_loss, val_decode[4]) + s
            )

        # using checkpoints to perform early-stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'checkpoints/{}/model_{}'.format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
    return model_path
