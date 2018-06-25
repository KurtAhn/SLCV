from __init__ import *
import dataset as ds
from model import Model
import tensorflow as tf
try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    from tensorflow.contrib.data import TFRecordDataset
import numpy as np


NL = ds.LX_DIM
NC = cfg_net.get('nc', 2)
NH = cfg_net.get('nh', 64)
DH = cfg_net.get('dh', 6)
NA = ds.AX_DIM
NE = cfg_net.get('ne', 300)
DE = cfg_net.get('de', 2)
RP = 1e-5
DEVICE = cfg_net.get('device', 'cpu')
DEVICE = '/gpu:0' if DEVICE == 'gpu' else '/cpu:0'


class Encoder(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'ahn18.Encoder', **kwargs)

    def _create(self, **kwargs):
        init = lambda shape: tf.truncated_normal(shape, stddev=0.1)

        with tf.device(DEVICE):
            with tf.name_scope(self.name) as scope:
                tf.placeholder('float', [None, None, NE], name='e')
                tf.placeholder('int64', [None], name='n')
                tf.placeholder('float', [None, NC], name='s_')

                self._optimizer = tf.train.AdamOptimizer()

                cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(NE)] * DE,
                    state_is_tuple=True)
                y, q = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cells,
                    cell_bw=cells,
                    inputs=self['e'],
                    sequence_length=self['n'],
                    dtype='float',
                    time_major=False,
                    scope=scope
                )

                tf.Variable(init([NE*2, NC]), name='P')
                tf.matmul(tf.concat([q[0][-1].h, q[1][-1].h], axis=1), self['P'], name='s')
                tf.reduce_mean(tf.square(self['s'] - self['s_']), name='j')
                self.optimizer.minimize(self['j'], name='o')

    def encode(self, tokens, lengths, targets, train=False):
        session = tf.get_default_session()
        if train:
            return session.run(
                [self['s'], self['j'], self['o']],
                feed_dict={
                    self['e']: tokens,
                    self['n']: lengths,
                    self['s_']: targets
                }
            )[:-1]
        elif targets is not None:
            return session.run(
                [self['s'], self['j']],
                feed_dict={
                    self['e']: tokens,
                    self['n']: lengths,
                    self['s_']: targets
                }
            )
        else:
            return session.run(
                [self['s']],
                feed_dict={
                    self['e']: tokens,
                    self['n']: lengths
                }
            )

    @property
    def optimizer(self):
        return self._optimizer


class Synthesizer(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'ahn18.Synthesizer', **kwargs)

    def _create(self, **kwargs):
        init = lambda shape: tf.truncated_normal(shape, stddev=0.1)

        with tf.device(DEVICE):
            with tf.name_scope(self.name) as scope:
                tf.placeholder('float', [None, NL], name='l')
                tf.placeholder('float', [None, NC], name='c')
                tf.placeholder('float', [None, NA], name='a_')

                self._optimizer = tf.train.AdamOptimizer()

                tf.Variable(init([NL+NC, NH]), name='W0')
                for d in range(1, DH-1):
                    tf.Variable(init([NH, NH]), name='W{}'.format(d))
                tf.Variable(init([NH, NA]), name='W{}'.format(DH-1))

                for d in range(DH-1):
                    tf.Variable(init([1, NH]), name='b{}'.format(d))
                tf.Variable(init([1, NA]), name='b{}'.format(DH-1))

                h = tf.concat([self['l'], self['c']], axis=1)
                for d in range(DH-1):
                    h = tf.nn.tanh(tf.add(tf.matmul(h, self['W{}'.format(d)]),
                                          self['b{}'.format(d)]))
                tf.add(tf.matmul(h, self['W{}'.format(DH-1)]),
                       self['b{}'.format(DH-1)],
                       name='a')

                j = sum([RP * tf.nn.l2_loss(self['W{}'.format(d)])
                         for d in range(DH)],
                        tf.reduce_mean(tf.square(self['a'] - self['a_'])))
                tf.identity(j, name='j')
                self.optimizer.minimize(self['j'], name='o')

    def synthesize(self, linguistics, controls, targets, train=False):
        session = tf.get_default_session()
        if train:
            return session.run(
                [self['a'], self['j'], self['o']],
                feed_dict={
                    self['l']: linguistics,
                    self['c']: controls,
                    self['a_']: targets
                }
            )[:2]
        elif targets is not None:
            return session.run(
                [self['a'], self['j']],
                feed_dict={
                    self['l']: linguistics,
                    self['c']: controls,
                    self['a_']: targets
                }
            )
        else:
            return session.run(
                [self['a']],
                feed_dict={
                    self['l']: linguistics,
                    self['c']: controls
                }
            )

    @property
    def optimizer(self):
        return self._optimizer


def load_encoder_dataset(sentences, oracle=None):
    indices = {s: n for n, s in enumerate(sentences)}
    dataset = TFRecordDataset([path.join(DSSDIR, sentence+'.tfr')
                               for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'e': tf.FixedLenSequenceFeature([NE],
                                                        tf.float32,
                                                        allow_missing=True),
                        'n': tf.FixedLenFeature([], tf.int64)
                    }
                )
        )

    if oracle is None:
        return dataset.map(lambda feature: (feature['e'], feature['n']))
    else:
        with open(path.join(ORCWDIR, oracle+'.orc'), 'rb') as f:
            oracle = np.load(f)
        return dataset.map(lambda feature: \
            (feature['e'],
             feature['n'],
             tf.py_func(
                lambda s: oracle[indices[s.decode('ascii')],:],
                [feature['s']],
                tf.float32
             ))
        )


def load_synthesizer_dataset(sentences, oracle=None):
    indices = {s: n for n, s in enumerate(sentences)}
    dataset = TFRecordDataset([path.join(DSATDIR, sentence+'.tfr')
                               for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'w': tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([ds.LX_DIM], tf.float32),
                        'a': tf.FixedLenFeature([ds.AX_DIM], tf.float32)
                    }
                )
        )

    if oracle is None:
        return dataset.map(lambda feature: (feature['l'], feature['a']))
    else:
        with open(path.join(ORCADIR, oracle+'.orc'), 'rb') as f:
            oracle = np.load(f)
        return dataset.map(lambda feature: \
            (feature['l'],
             tf.py_func(
                lambda s: oracle[indices[s.decode('ascii')],:],
                [feature['s']],
                tf.float32
             ),
             feature['a'])
        )
