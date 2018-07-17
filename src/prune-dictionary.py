#!/usr/bin/env python
from __init__ import load_config
from os import path
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-w', '--whole', dest='whole', required=True)
    p.add_argument('-p', '--pruned', dest='pruned', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    print2('Loading whole vocabulary')
    with open(path.join(EMBDIR, a.whole+'.vcb')) as f:
        whole_vocab = [l.rstrip() for l in f]
    whole_vocab = {t: n for n, t in enumerate(whole_vocab)}

    print2('Loading embedding')
    with open(path.join(EMBDIR, a.whole+'.dim')) as f:
        whole_embed_shape = tuple(int(t) for t in next(f).rstrip().split())
    whole_embed = np.memmap(path.join(EMBDIR, a.whole+'.emb'),
                            mode='r', dtype='float', shape=whole_embed_shape)

    print2('Pruning vocabulary')
    partial_vocab = set()
    for sentence in sentences:
        with open(path.join(TOKDIR, sentence+'.txt')) as f:
            partial_vocab |= {t for t in next(f).rstrip().split()}
    partial_vocab = sorted(list(partial_vocab))
    with open(path.join(EMBDIR, a.pruned+'.vcb'), 'w') as f:
        for w in partial_vocab:
            f.write(w+'\n')

    partial_embed_shape = (len(partial_vocab), whole_embed_shape[1])
    with open(path.join(EMBDIR, a.pruned+'.dim'), 'w') as f:
        f.write('{} {}'.format(*partial_embed_shape))

    partial_embed = np.memmap(path.join(EMBDIR, a.pruned+'.emb'),
                              mode='w+', dtype='float', shape=partial_embed_shape)
    for n, w in enumerate(partial_vocab):
        partial_embed[n,:] = whole_embed[whole_vocab[w],:]
