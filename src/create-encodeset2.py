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
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    F = tf.train.Feature
    FF = tf.train.Features
    E = tf.train.Example
    BL = tf.train.BytesList
    FL = tf.train.FloatList
    IL = tf.train.Int64List
    W = tf.python_io.TFRecordWriter

    for sentence in sentences:
        with open(path.join(TOKDIR, sentence+'.txt')) as f:
            words = [w for w in next(f).rstrip().split()]

        with W(path.join(DSSDIR, sentence+'.tfr')) as writer:
            features = {
                's': F(bytes_list=BL(value=[bytes(sentence, encoding='ascii')])),
                'n': F(int64_list=IL(value=[len(words)])),
                'w': F(bytes_list=BL(value=[bytes(w, encoding='ascii')
                                            for w in words]))
            }
            example = E(features=FF(feature=features))
            writer.write(example.SerializeToString())

        print1(sentence)
        flush1()
