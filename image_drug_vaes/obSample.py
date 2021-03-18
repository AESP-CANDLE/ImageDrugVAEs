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

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', required=True, help='saved trained model file (.pt)', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    loaded = torch.load(args.m, map_location='cpu')

    vocab = pickle.load(open(config['vocab_file'], "rb"))
    vocab = {k: v for v, k in enumerate(vocab)}

    train_data = MoleLoader(pd.read_csv("moses/data/val.csv"), vocab, max_len=config['vocab_max_len'],
                            start_char=config['start_char'], end_char=config['end_char'])

    train_loader_food = torch.utils.data.DataLoader(
        train_data,
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
    decoder.load_state_dict(loaded['decoder_state_dict'])
    decoder = decoder.to(device)

    ####

    ### Set up encoder

    encoder = Encoder()
    encoder.fine_tune(config['train_encoder'])
    encoder.load_state_dict(loaded['encoder_state_dict'])
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    def test(epoch):
        print("Epoch {}: batch_size {}".format(epoch, config['batch_size']))
        decoder.eval()  # train mode (dropout and batchnorm is used)
        encoder.eval()
        total_losses = AverageMeter()  # loss (per word decoded)
        total_per_char_acc = AverageMeter()
        total_string_acc = AverageMeter()
        total_editDist = AverageMeter()

        for batch_idx, (embed, data, embedlen) in enumerate(train_loader_food):
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

            print(f'acc_c: {acc_c}, edit_distance: {1.0 if s1 == s2 else 0.0}')
            return total_losses.avg

    with torch.no_grad():
        test(1)
