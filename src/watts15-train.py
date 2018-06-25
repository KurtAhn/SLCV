#!/usr/bin/env python
from __init__ import load_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path, mkdir
import shutil
import numpy as np
from random import Random
from argparse import ArgumentParser


# def memory():
#     import os, psutil
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     use = py.memory_info()[0]/2.0**32
#     print2('memory use: ', use)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('--batch', dest='batch', type=int, default=256)
    p.add_argument('--size', dest='size', type=int, default=None)
    p.add_argument('--penalty', dest='penalty', type=float, default=1e-5)
    p.add_argument('--split', dest='split', type=float, default=0.95)
    p.add_argument('--rate', dest='rate', type=float, default=0.001)
    p.add_argument('--keep', dest='keep', type=float, default=1.0)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from watts15 import *
    import util

    mdldir = path.join(MDLWDIR, a.model)
    try:
        mkdir(mdldir)
    except FileExistsError:
        pass
    shutil.copy2(a.config, mdldir)

    log_path = path.join(mdldir, 'log.txt')
    with open(log_path, 'a') as f:
        f.write('---------------------------\n')
        f.write('NC: {}\n'.format(NC))
        f.write('NH: {}\n'.format(NH))
        f.write('DH: {}\n'.format(DH))
        f.write('batch: {}\n'.format(a.batch))
        f.write('penalty: {}\n'.format(a.penalty))
        f.write('rate: {}\n'.format(a.rate))
        f.write('keep: {}\n'.format(a.keep))
        f.flush()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    if a.size is None:
        print2('Counting examples')
        size = ds.count_examples([path.join(DSATDIR, s+'.tfr')
                               for s in sentences])
    else:
        size = a.size
    t_size = int(a.split * size / a.batch)

    with open(log_path, 'a') as f:
        f.write('size: {}\n'.format(size))
        f.write('split: {}\n'.format(a.split))
        f.write('---------------------------\n')

    data = load_dataset(Random(SEED).sample(sentences, len(sentences)))\
           .batch(a.batch)\
           .shuffle(buffer_size=100,
                    seed=SEED,
                    reshuffle_each_iteration=False)\
           .make_initializable_iterator()
    example = data.get_next()
    print2('Dataset created')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config).as_default() as session:
        if a.epoch == 0:
            model = Trainer(sentences=sentences)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
        else:
            model = Trainer(mdldir=mdldir, epoch=a.epoch)
            session.run(tf.tables_initializer())
        print2('Model created')

        saver = tf.train.Saver(max_to_keep=0)

        epoch = a.epoch+1
        v_loss = None
        while True:
            t_report = util.Report(epoch, mode='t')
            session.run(data.initializer)
            while t_report.iterations < t_size:
                try:
                    out, loss = model.train(*session.run(example),
                                            train=True,
                                            l2_penalty=a.penalty,
                                            learning_rate=a.rate,
                                            keep_prob=a.keep)
                    t_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            v_report = util.Report(epoch, mode='v')
            while True:
                try:
                    out, loss = model.train(*session.run(example),
                                            train=False)
                    v_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            with open(log_path, 'a') as f:
                f.write("{},{:.3e},{:.3e}\n"\
                        .format(epoch, t_report.avg_loss, v_report.avg_loss))
                f.flush()

            model.save(saver, mdldir, epoch)
            epoch += 1

            v_loss = v_report.avg_loss
