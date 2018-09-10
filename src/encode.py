#!/usr/bin/env python
from __init__ import load_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path, mkdir
import numpy as np
import matplotlib.pyplot as pyplot
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, required=True)
    p.add_argument('-l', '--limits', dest='limits', type=float, nargs=4, default=None)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    from ahn18 import *

    outdir = path.join(ORCADIR, a.model)
    try:
        mkdir(outdir)
    except FileExistsError:
        pass

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    data = load_encoder_dataset2(sentences)\
           .padded_batch(1, padded_shapes=([None],[]))\
           .make_initializable_iterator()
    example = data.get_next()

    with tf.Session().as_default() as session:
        model = Encoder(mdldir=path.join(MDLEDIR, a.model), epoch=a.epoch)
        session.run(tf.tables_initializer())
        session.run(data.initializer)

        # oracle = []
        while True:
            try:
                control, = model.encode(*session.run(example), None,
                                        train=False)
                print1(*control[0])
                # oracle.append(control)
            except tf.errors.OutOfRangeError:
                break

    # oracle = np.concatenate(oracle, axis=0)

    # mean = np.mean(oracle, axis=0)
    # distance = np.linalg.norm(oracle-mean, axis=1)
    # print2(list(map(lambda e: (sentences[e[0]], e[1]),
    #                 sorted(enumerate(distance), key=lambda e: e[1], reverse=True)))[:10])
    #
    # with open(path.join(outdir, '{}.orc'.format(a.epoch)), 'wb') as f:
    #     np.save(f, oracle)
    #
    # pyplot.scatter(*zip(*oracle.reshape(-1,2)), s=1)
    # if a.limits is not None:
    #     pyplot.xlim(a.limits[:2])
    #     pyplot.ylim(a.limits[2:])
    # pyplot.savefig(path.join(outdir, '{}.pdf'.format(a.epoch)))
