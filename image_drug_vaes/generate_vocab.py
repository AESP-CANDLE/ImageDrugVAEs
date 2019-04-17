from config import *
config = Config().get_global_config()

import pandas as pd
import pickle

df = pd.concat([pd.read_csv(config['train_data']), pd.read_csv(config['val_data'])], axis=0)


def generate_vocab(df):
    s = set(' ')
    for i, row in df.iterrows():
        s = s.union(row.iloc[0])
    print(s)
    return list(s)


vocab = generate_vocab(df)
vocab.append('!')
vocab.append('?')
vocab.insert(0, ' ')

with open(config['vocab_file'], 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

