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
    from ahn18 import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    data = load_encoder_dataset(sentences)\
           .padded_batch(1, padded_shapes=([None,NE],[]))\
           .make_initializable_iterator()
    example = data.get_next()
    print2('Dataset created')

    with tf.Session().as_default() as session:
        model = Encoder(mdldir=path.join(MDLAEDIR, a.model), epoch=a.epoch)
        session.run(tf.global_variables_initializer())
        session.run(data.initializer)

        oracle = []
        while True:
            try:
                control, = model.predict(*session.run(example), None,
                                         train=False)
                oracle.append(control)
            except tf.errors.OutOfRangeError:
                break

    with open(path.join(ORCADIR, '{}-{}.orc'.format(a.model, a.epoch)), 'wb') as f:
        np.save(f, np.concatenate(oracle, axis=0))
