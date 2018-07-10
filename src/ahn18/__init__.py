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
NC = ds.NC
# NC = cfg_net.get('nc', 2)
# NH = cfg_enc.get('nh', 64)
# DH = cfg_enc.get('dh', 6)
NA = ds.AX_DIM
NE = cfg_enc.get('ne', 300)
NP = cfg_enc.get('np', 300)
DP = cfg_enc.get('dp', 0)
DR = cfg_enc.get('dr', 1)
# RP = 1e-5
USE_LSTM = cfg_enc.get('lstm', True)
DEVICE = cfg_enc.get('device', 'cpu')
DEVICE = '/gpu:0' if DEVICE == 'gpu' else '/cpu:0'


class Encoder(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'Encoder', **kwargs)

    def _create(self, **kwargs):
        winit = tf.contrib.layers.xavier_initializer()
        wfunc = tf.nn.tanh
        rinit = tf.contrib.layers.variance_scaling_initializer()
        rfunc = tf.nn.relu
        pinit = rinit #lambda shape: tf.random_normal(shape, stddev=0.1)

        with tf.device(DEVICE):
            with tf.name_scope(self.name) as scope:
                tf.placeholder('float', [None, None, NE], name='e')
                tf.placeholder('int64', [None], name='n')
                tf.placeholder('float', [None, NC], name='s_')

                self._optimizer = tf.train.AdamOptimizer(
                    learning_rate=tf.placeholder('float', name='learning_rate')
                )

                if DP > 0:
                    tf.Variable(winit([NE,NP]), name='W0')
                    tf.Variable(winit([1,NP]), name='b0')
                    h = tf.scan(lambda a, x: wfunc(x @ self['W0'] + self['b0']), self['e'])
                    # h = wfunc(self['e'] @ self['W0'] + self['b0'])
                    if DP > 1:
                        for d in range(1,DP):
                            Wd = 'W{}'.format(d)
                            bd = 'b{}'.format(d)
                            tf.Variable(winit([NP,NP]), name=Wd)
                            tf.Variable(winit([1,NP]), name=bd)
                            h = tf.scan(lambda a, x: wfunc(x @ self[Wd] + self[bd]), self['e'])
                            # h = wfunc(h @ self[Wd] + self[bd])
                else:
                    h = self['e']

                cell_type = tf.nn.rnn_cell.BasicLSTMCell if USE_LSTM else \
                            tf.nn.rnn_cell.GRUCell

                with tf.variable_scope('brnn', initializer=rinit):
                    cells = tf.nn.rnn_cell.MultiRNNCell(
                        [cell_type(num_units=NP, activation=rfunc) for d in range(DR)],
                        state_is_tuple=True
                    )

                    y, q = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cells,
                        cell_bw=cells,
                        inputs=h,
                        sequence_length=self['n'],
                        dtype='float',
                        time_major=False
                    )

                if USE_LSTM:
                    tf.concat([q[0][-1].h, q[1][-1].h], axis=1, name='q')
                else:
                    tf.concat([q[0][-1], q[1][-1]], axis=1, name='q')

                tf.Variable(pinit([2*NP, NC]), name='P')
                tf.matmul(self['q'], self['P'], name='s')
                tf.reduce_mean(tf.square(self['s'] - self['s_']), name='j')
                self.optimizer.minimize(self['j'], name='o')

    def encode(self, tokens, lengths, targets, train=False, **kwargs):
        learning_rate = kwargs.get('learning_rate', 0.001)
        session = tf.get_default_session()
        if train:
            return session.run(
                [self['s'], self['j'], self['o']],
                feed_dict={
                    self['e']: tokens,
                    self['n']: lengths,
                    self['s_']: targets,
                    self['learning_rate']: learning_rate
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


# class Synthesizer(Model):
#     def __init__(self, **kwargs):
#         Model.__init__(self, 'ahn18.Synthesizer', **kwargs)
#
#     def _create(self, **kwargs):
#         init = tf.contrib.layers.xavier_initializer()
#         func = tf.nn.tanh
#
#         with tf.device(DEVICE):
#             with tf.name_scope(self.name) as scope:
#                 tf.placeholder('float', [None, NL], name='l')
#                 tf.placeholder('float', [None, NC], name='c')
#                 tf.placeholder('float', [None, NA], name='a_')
#
#                 tf.placeholder('float', name='l2_penalty')
#                 tf.placeholder('float', name='keep_prob')
#                 self._optimizer = tf.train.AdamOptimizer(
#                     learning_rate=tf.placeholder('float', name='learning_rate')
#                 )
#
#                 tf.Variable(init([NL+NC, NH]), name='W0')
#                 for d in range(1, DH-1):
#                     tf.Variable(init([NH, NH]), name='W{}'.format(d))
#                 tf.Variable(init([NH, NA]), name='W{}'.format(DH-1))
#
#                 for d in range(DH-1):
#                     tf.Variable(init([1, NH]), name='b{}'.format(d))
#                 tf.Variable(init([1, NA]), name='b{}'.format(DH-1))
#
#                 h = tf.concat([self['l'], self['c']], axis=1)
#                 for d in range(DH-1):
#                     h = func(tf.add(tf.matmul(h, self['W{}'.format(d)]),
#                                     self['b{}'.format(d)]))
#                     h = tf.nn.dropout(h, self['keep_prob'])
#                 tf.add(tf.matmul(h, self['W{}'.format(DH-1)]),
#                        self['b{}'.format(DH-1)],
#                        name='a')
#
#                 # j = sum([self['l2_penalty'] * tf.nn.l2_loss(self['W{}'.format(d)])
#                 #          for d in range(DH)],
#                 #         tf.reduce_mean(tf.square(self['a'] - self['a_'])))
#                 # tf.identity(j, name='j')
#                 j = tf.reduce_mean(tf.abs(self['a'] - self['a_']))
#                 tf.add(j, tf.add_n([tf.nn.l2_loss(self['W{}'.format(d)])
#                                     for d in range(DH)]) * self['l2_penalty'],
#                        name='j')
#                 self.optimizer.minimize(self['j'], name='o')
#
#     def synth(self, linguistics, controls, targets, train=False, **kwargs):
#         l2_penalty = kwargs.get('l2_penalty', 1e-5)
#         learning_rate = kwargs.get('learning_rate', 0.001)
#         keep_prob = kwargs.get('keep_prob', 1.0)
#
#         session = tf.get_default_session()
#         if train:
#             return session.run(
#                 [self['a'], self['j'], self['o']],
#                 feed_dict={
#                     self['l']: linguistics,
#                     self['c']: controls,
#                     self['a_']: targets,
#                     self['l2_penalty']: l2_penalty,
#                     self['learning_rate']: learning_rate,
#                     self['keep_prob']: keep_prob
#                 }
#             )[:2]
#         elif targets is not None:
#             return session.run(
#                 [self['a'], self['j']],
#                 feed_dict={
#                     self['l']: linguistics,
#                     self['c']: controls,
#                     self['a_']: targets,
#                     self['l2_penalty']: l2_penalty,
#                     self['keep_prob']: 1.0
#                 }
#             )
#         else:
#             return session.run(
#                 [self['a']],
#                 feed_dict={
#                     self['l']: linguistics,
#                     self['c']: controls,
#                     self['keep_prob']: 1.0
#                 }
#             )
#
#     @property
#     def optimizer(self):
#         return self._optimizer


def load_encoder_dataset(sentences, oracle=None):
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
        indices = {s: n for n, s in enumerate(sentences)}
        return dataset.map(lambda feature: \
            (feature['e'],
             feature['n'],
             tf.py_func(
                lambda s: oracle[indices[s.decode('ascii')],:].reshape(NC),
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
                        # 'w': tf.FixedLenFeature([], tf.string),
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
