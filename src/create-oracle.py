#!/usr/bin/python3
from __init__ import *
from watts15 import *
from watts15.dnn import *
from argparse import ArgumentParser
import util
import numpy as np
import matplotlib.pyplot as pyplot


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    with tf.Session().as_default() as session:
        n1 = SLCV1(mdldir=path.join(MDLDIR, a.model))
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        c = [n1.embed(s) for s in sentences]

    pyplot.scatter(*zip(*c), s=1)
    pyplot.show()