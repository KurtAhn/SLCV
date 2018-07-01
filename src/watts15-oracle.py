#!/usr/bin/env python
from __init__ import load_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as pyplot
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    from watts15 import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    with tf.Session().as_default() as session:
        model = Trainer(mdldir=path.join(MDLWDIR, a.model), epoch=a.epoch)
        session.run(tf.tables_initializer())

        oracle = []
        for s in sentences:
            control, = model.embed([s])
            oracle.append(control)

    oracle = np.concatenate(oracle, axis=0)

    print2(np.var(oracle))

    mean = np.mean(oracle, axis=0)
    distance = np.linalg.norm(oracle-mean, axis=1)
    extreme = list(map(lambda e: (sentences[e[0]], e[1]),
                       sorted(enumerate(distance), key=lambda e: e[1], reverse=True)))[:10]
    # for e in extreme:
    #     print2(e[0], oracle[sentences.index(e[0])])

    with open(path.join(ORCWDIR, '{}-{}.orc'.format(a.model, a.epoch)), 'wb') as f:
        np.save(f, oracle)

    pyplot.scatter(*zip(*oracle.reshape(-1,2)), s=1)
    pyplot.savefig(path.join(ORCWDIR, '{}-{}.pdf'.format(a.model, a.epoch)))
