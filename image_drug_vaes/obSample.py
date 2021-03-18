from config import *

config = Config().get_im_smi_config()

import datetime
import pickle

import pandas as pd
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence

from models.gridLSTM import GridLSTMDecoderWithAttention, Encoder
from utils.DataLoader import MoleLoader
from utils.utils import clip_gradient, AverageMeter, levenshteinDistance, CometHolder


if __name__ == '__main__':
    vocab = pickle.load(open(config['vocab_file'], "rb"))
    vocab = {k: v for v, k in enumerate(vocab)}

    train_data = MoleLoader(pd.read_csv("moses/data/train.csv"), vocab, max_len=config['vocab_max_len'],
                            start_char=config['start_char'], end_char=config['end_char'])
    val_data = MoleLoader(pd.read_csv("moses/data/test.csv"), vocab, max_len=config['vocab_max_len'],
                          start_char=config['start_char'], end_char=config['end_char'])

    train_loader_food = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True, drop_last=True, **config['data_loader_kwargs'])
    val_loader_food = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=True, drop_last=True, **config['data_loader_kwargs'])

    charset = train_data.charset
    embedding_width = config['vocab_max_len']
    embedding_size = len(vocab)

    device = torch.device("cuda" if config['cuda'] else "cpu")
    kwargs = config['data_loader_kwargs'] if config['cuda'] else {}

    ### Set up decoder
    decoder = GridLSTMDecoderWithAttention(attention_dim=config['attention_dim'],
                                           embed_dim=config['emb_dim'],
                                           decoder_dim=config['decoder_dim'],
                                           vocab_size=len(vocab),
                                           encoder_dim=config['encoder_dim'],
                                           dropout=config['dropout'],
                                           device=device)

    decoder.fine_tune_embeddings(True)

    decoder = decoder.to(device)

    decoder_optimizer = None
    if config['dec_optimizer'] == 'adam':
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             **config['dec_optimizer_kwargs'])
    elif config['dec_optimizer'] == 'sgd':
        decoder_optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            **config['dec_optimizer_kwargs'])
    else:
        raise ValueError("I'm going to bet in config.py an optimizer isn't 'adam' or 'sgd'! Fix it.")

    decoder_sched = None
    if config['dec_lr_scheduling'] == 'annealing':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, **config['dec_lr_scheduling_kwargs'])
    elif config['dec_lr_scheduling'] == 'reduce_on_plateau':
        raise ValueError("The config says I will implement this. But I have not. Stick to annealing.")

    ####

    ### Set up encoder

    encoder = Encoder()
    encoder.fine_tune(config['train_encoder'])

    encoder = encoder.to(device)

    encoder_optimizer = None
    if config['enc_optimizer'] == 'adam':
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             **config['enc_optimizer_kwargs'])
    elif config['enc_optimizer'] == 'sgd':
        encoder_optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            **config['enc_optimizer_kwargs'])
    else:
        raise ValueError("I'm going to bet in config.py an optimizer isn't 'adam' or 'sgd'! Fix it.")

    encoder_sched = None
    if config['enc_lr_scheduling'] == 'annealing':
        encoder_sched = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, **config['enc_lr_scheduling_kwargs'])
    elif config['enc_lr_scheduling'] == 'reduce_on_plateau':
        raise ValueError("The config says I will implement this. But I have not. Stick to annealing.")

    criterion = nn.CrossEntropyLoss()
    if config['distributed']:
        criterion = criterion.to(device)


    def test(epoch):
        print("Epoch {}: batch_size {}".format(epoch, config['batch_size']))
        decoder.test()  # train mode (dropout and batchnorm is used)
        encoder.test()
        total_losses = AverageMeter()  # loss (per word decoded)
        total_per_char_acc = AverageMeter()
        total_string_acc = AverageMeter()
        total_editDist = AverageMeter()

        for batch_idx, (embed, data, embedlen) in enumerate(val_loader_food):
            imgs_orig = data.float()
            imgs_orig = imgs_orig.to(device)
            caps = embed.to(device)
            caplens = embedlen.to(device).view(-1, 1)

            # Forward prop.
            imgs_vae = encoder(imgs_orig)

            imgs_vae = imgs_vae.to(device)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_vae, caps, caplens,
                                                                            teacher_forcing=bool(epoch < 3))

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            imgs_vae = imgs_vae[sort_ind]
            imgs_orig = imgs_orig[sort_ind]

            print('imgs_vae_shape: ', imgs_vae.shape)
            print(imgs_orig.shape)

            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            targets_copy = targets.clone()

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            total_losses.update(loss.item(), sum(decode_lengths))

            # acc_per_c
            acc_c = torch.max(scores.data, dim=1)[1].eq(targets.data).sum().item() / float(targets.data.shape[0])
            total_per_char_acc.update(acc_c)

            # acc_per_string
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.cpu().numpy()
            targets_copy = targets_copy.cpu().numpy()
            for i in range(preds.shape[0]):
                s1 = preds[i, ...]
                s2 = targets_copy[i, ...]
                s1 = "".join([charset[chars] for chars in s1]).strip()
                s2 = "".join([charset[chars] for chars in s2]).strip()
                total_string_acc.update(1.0 if s1 == s2 else 0.0)
                total_editDist.update(levenshteinDistance(s1, s2))

            return total_losses.avg


    test()
