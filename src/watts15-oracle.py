#!/usr/bin/env python
from __init__ import load_config
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
        model = SLCV1(mdldir=path.join(MDLDIR, a.model))
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        control = [model.embed(s) for s in sentences]

    with open(path.join(ORCWDIR, '{}-{}.orc'.format(a.model, a.epoch)), 'wb') as f:
        np.save(f, control)
