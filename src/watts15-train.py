#!/usr/bin/python3
from __init__ import *
import acoustic as ax
import dataset as ds
from watts15 import *
from watts15.dnn import *
from os import path, mkdir
from argparse import ArgumentParser
import util
import numpy as np


if __name__ == '__main__':
    p = ArgumentParser()
    # p.add_argument('-m', '--model', dest='model', required=True)
    # p.add_argument('-e', '--epoch', dest='epoch', type=int, default=None)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-o', '--outdir', dest='outdir', required=True)
    p.add_argument('-n', '--ndataset', dest='ndataset', type=int, default=None)
    p.add_argument('-b', '--nbatch', dest='nbatch', type=int, default=256)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]
    records = [path.join(TRNDIR, s+'.tfr') for s in sentences]

    try:
        mkdir(path.join(MDLDIR, a.outdir))
    except FileExistsError:
        pass

    with tf.Session().as_default() as session:
        model = SLCV1(sentences=sentences,
                      nl=NL,
                      nc=NC,
                      nh=NH,
                      na=NA,
                      dp=DP,
                      rp=RP)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=0)

        eprint('Model created')

        n = a.ndataset or ds.count_examples(records)
        nt = int(n * 0.8)

        eprint('Example count', n)

        dataset = ds.load_trainset(records)\
            .shuffle(buffer_size=100000,
                     reshuffle_each_iteration=False)\
            .batch(a.nbatch)

        eprint('Dataset created')

        epoch = 1
        dev_loss = None
        while True:
            eprint('Training epoch', epoch)
            trn_report = util.Report(epoch, mode='t')
            example = dataset.make_one_shot_iterator().get_next()
            count = 0
            while True:
                try:
                    loss = model.train(*session.run(example))
                    trn_report.report(loss)
                    count += a.nbatch
                    if count >= nt:
                        break
                except tf.errors.OutOfRangeError:
                    break
            trn_report.flush()

            dev_report = util.Report(epoch, mode='d')
            #dev_example = dev_data.make_one_shot_iterator().get_next()
            while True:
                try:
                    loss, out = model.predict(*session.run(example))
                    dev_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            dev_report.flush()

            if dev_loss is not None and \
               dev_loss < dev_report.total_loss:
                break
            dev_loss = dev_report.total_loss

            if epoch == 1:
                saver.save(session,
                           path.join(MDLDIR, a.outdir, '_'),
                           write_meta_graph=True)
            else:
                saver.save(session,
                           path.join(MDLDIR, a.outdir, '_'),
                           global_step=epoch,
                           write_meta_graph=False)
            epoch += 1
