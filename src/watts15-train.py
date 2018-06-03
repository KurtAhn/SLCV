#!/usr/bin/env python
from __init__ import *
import acoustic as ax
import dataset as ds
from watts15 import *
from watts15.dnn import *
from os import path, mkdir
from argparse import ArgumentParser
import util
import numpy as np
from random import shuffle


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-n', '--ndataset', dest='ndataset', type=int, default=None)
    p.add_argument('-b', '--nbatch', dest='nbatch', type=int, default=256)
    a = p.parse_args()

    try:
        mkdir(path.join(MDLDIR, a.model))
    except FileExistsError:
        pass

    log_path = path.join(MDLDIR, a.model, 'log.txt')
    with open(log_path, 'a') as f:
        f.write('----MODEL CONFIGURATION----\n')
        f.write('Control: {}\n'.format(NC))
        f.write('Depth: {}\n'.format(DP))
        f.write('Nodes per layer: {}\n'.format(NH))
        f.write('---------------------------\n')
        f.flush()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]
    records = [path.join(TRNDIR, s+'.tfr') for s in sentences]
    shuffle(records)

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config).as_default() as session:
        if a.epoch == 0:
            model = SLCV1(sentences=sentences,
                          nl=NL,
                          nc=NC,
                          nh=NH,
                          na=NA,
                          dp=DP,
                          rp=RP)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
        else:
            model = SLCV1(mdldir=path.join(MDLDIR, a.model), epoch=a.epoch)
            #session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=0)

        print2('Model created')

        n = a.ndataset or ds.count_examples(records)
        nt = int(n * 0.95)

        print2('Example count', n)

        dataset = ds.load_trainset(records)\
            .shuffle(buffer_size=100000,
                     reshuffle_each_iteration=False)\
            .batch(a.nbatch)

        print2('Dataset created')

        epoch = a.epoch+1
        dev_loss = None
        while True:
            print2('Training epoch', epoch)
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

            with open(log_path, 'a') as f:
                f.write("{},{:.3e},{:.3e}\n"\
                        .format(epoch, trn_report.avg_loss, dev_report.avg_loss))
                f.flush()

            if epoch == 1:
                tf.train.export_meta_graph(
                    filename=path.join(MDLDIR, a.model, '_.meta')
                )
            saver.save(session,
                       path.join(MDLDIR, a.model, '_'),
                       global_step=epoch,
                       write_meta_graph=False)
            epoch += 1

            if epoch > 15 and \
               dev_loss is not None and \
               dev_loss < dev_report.avg_loss:
                break
            dev_loss = dev_report.avg_loss
