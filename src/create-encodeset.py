#!/usr/bin/env python
from __init__ import load_config
from os import path
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-v', '--vocab', dest='vocab', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    print2('Loading vocabulary')
    with open(path.join(VOCDIR, a.vocab+'.txt')) as f:
        vocabulary = [l.rstrip() for l in f]
    vocabulary = {t: n for n, t in enumerate(vocabulary)}

    print2('Loading embedding')
    with open(path.join(EMBDIR, a.vocab+'.dim.txt')) as f:
        embedding_shape = tuple(int(t) for t in next(f).rstrip().split())
    embedding = np.memmap(path.join(EMBDIR, a.vocab+'.emb'),
                          mode='r', dtype='float', shape=embedding_shape)

    F = tf.train.Feature
    FF = tf.train.Features
    E = tf.train.Example
    BL = tf.train.BytesList
    FL = tf.train.FloatList
    IL = tf.train.Int64List
    W = tf.python_io.TFRecordWriter

    for sentence in sentences:
        with open(path.join(TOKDIR, sentence+'.txt')) as f:
            tokens = np.vstack([embedding[vocabulary.get(t, -1),:]
                                for t in next(f).rstrip().split()])

        with W(path.join(ENCDIR, sentence+'.tfr')) as writer:
            features = {
                's': F(bytes_list=BL(value=[bytes(sentence, encoding='ascii')])),
                'n': F(int64_list=IL(value=[tokens.shape[0]])),
                'e': F(float_list=FL(value=tokens.reshape(-1)))
            }
            example = E(features=FF(feature=features))
            writer.write(example.SerializeToString())

        print1(sentence)
        flush1()
