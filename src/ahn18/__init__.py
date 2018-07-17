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
NA = ds.AX_DIM
NE = cfg_enc.get('ne', 300)
NF = cfg_enc.get('nf', 300)
DF = cfg_enc.get('df', 1)
NR = cfg_enc.get('nr', 300)
DR = cfg_enc.get('dr', 1)
USE_LSTM = cfg_enc.get('lstm', True)
DEVICE = cfg_enc.get('device', 'cpu')
DEVICE = '/gpu:0' if DEVICE == 'gpu' else '/cpu:0'


class Encoder(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'Encoder', **kwargs)

    def _create(self, **kwargs):
        vocab = kwargs['vocab']
        embed = kwargs['embed']

        winit = tf.contrib.layers.xavier_initializer()
        wfunc = tf.nn.tanh
        binit = tf.zeros
        rinit = tf.contrib.layers.variance_scaling_initializer()
        rfunc = tf.nn.relu
        pinit = winit

        with tf.device(DEVICE):
            with tf.name_scope(self.name) as scope:
                tf.placeholder('string', [None, None], name='w')
                tf.placeholder('int32', [None], name='n')
                tf.placeholder('float', [None, NC], name='s_')

                tf.placeholder('float', name='keep_prob')
                self._optimizer = tf.train.AdamOptimizer(
                    tf.maximum(
                        tf.train.exponential_decay(
                            tf.placeholder('float', name='learning_rate'),
                            tf.Variable(0, trainable=False, dtype='int32', name='global_step'),
                            tf.placeholder('int32', name='dataset_size'),
                            tf.placeholder('float', name='decay_rate'),
                            False
                        ),
                        tf.placeholder('float', name='min_learning_rate')
                    )
                )
                tf.assign(self.global_step, self.global_step + 1)
                tf.placeholder('float', name='clip_threshold')

                table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(vocab),
                    # num_oov_buckets=1,
                    name='T'
                )

                tf.Variable(embed, dtype='float', name='E', trainable=True)
                if DF > 0:
                    tf.Variable(winit([NE,NF]), name='W0')
                    tf.Variable(binit([1,NF]), name='b0')
                    if DF > 1:
                        for d in range(1, DF):
                            tf.Variable(winit([NF,NF]), name='W{}'.format(d))
                            tf.Variable(winit([1,NF]), name='b{}'.format(d))
                def mf(x):
                    y = tf.nn.embedding_lookup(self.E, table.lookup(x))
                    for d in range(DF):
                        y = wfunc(y @ self['W{}'.format(d)] + self['b{}'.format(d)])
                    return y
                h = tf.map_fn(mf, self.w, dtype='float')

                cell_type = tf.nn.rnn_cell.BasicLSTMCell if USE_LSTM else \
                            tf.nn.rnn_cell.GRUCell

                self._brnn(cell_type, rinit, rfunc, pinit)(h)
                # self._cnn(rfunc, pinit)(h)

                tf.reduce_mean(tf.abs(self.s - self.s_), name='j')
                # self.optimizer.minimize(self.j, name='o')
                self.optimizer.apply_gradients(
                    [(tf.clip_by_value(g, -self.clip_threshold, self.clip_threshold), v)
                     for g, v in self.optimizer.compute_gradients(self.j)],
                    name='o'
                )

    def _rnn(self, cell_type, rinit, rfunc, pinit):
        def wrapper(h):
            with tf.variable_scope('rnn', initializer=rinit):
                cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.DropoutWrapper(
                        cell_type(num_units=NR, activation=rfunc),
                        output_keep_prob=self.keep_prob
                     )
                     for d in range(DR)],
                    state_is_tuple=True
                )

                y, q = tf.nn.dynamic_rnn(
                    cell=cells,
                    inputs=h,
                    sequence_length=self.n,
                    dtype='float',
                    time_major=False
                )

            if USE_LSTM:
                tf.identity(y[-1], name='q')
            else:
                # tf.identity(q[-1], name='q')
                tf.identity(y[-1], name='q')

            tf.Variable(pinit([NR, NC]), name='P')
            tf.matmul(self.q, self.P, name='s')
        return wrapper

    def _brnn(self, cell_type, rinit, rfunc, pinit):
        def wrapper(h):
            with tf.variable_scope('brnn', initializer=rinit):
                cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.DropoutWrapper(
                        cell_type(num_units=NR, activation=rfunc),
                        output_keep_prob=self.keep_prob
                     )
                     for d in range(DR)],
                    state_is_tuple=True
                )

                y, q = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cells,
                    cell_bw=cells,
                    inputs=h,
                    sequence_length=self.n,
                    dtype='float',
                    time_major=False
                )

            if USE_LSTM:
                tf.concat([y[0][-1], y[1][-1]], axis=1, name='q')
            else:
                tf.concat([y[0][-1], y[1][-1]], axis=1, name='q')

            tf.Variable(pinit([2*NR, NC]), name='P')
            tf.matmul(self.q, self.P, name='s')
        return wrapper

    def _cnn(self, activation, pinit):
        def wrapper(h):
            h = tf.expand_dims(h, -1)
            outputs = []
            n_max = 2
            for n in range(2, n_max+1):
                with tf.variable_scope('cnn{}'.format(n)):
                    l = h
                    for d in range(DR):
                        l = tf.nn.conv2d(
                            h,
                            tf.Variable(tf.truncated_normal([n, NE, 1, 1], stddev=0.01), name='W'),
                            strides=[1,1,1,1],
                            padding='SAME'
                        )
                        l = tf.nn.relu(
                            tf.nn.bias_add(l,
                                           tf.Variable(tf.zeros([1]), name='b')))
                    l = tf.nn.max_pool(
                        l,
                        ksize=[1, 46-n+1, 1, 1],
                        strides=[1,1,1,1],
                        padding='SAME'
                    )
                    outputs.append(l)
            q = tf.reshape(tf.stack(outputs), [-1, len(outputs)], name='q')
            tf.nn.dropout(q, self.keep_prob, name='q')
            tf.Variable(pinit([len(outputs), NC]), name='P')
            tf.matmul(self.q, self.P, name='s')
        return wrapper

    def encode(self, tokens, lengths, targets, train=False, **kwargs):
        learning_rate = kwargs.get('learning_rate', 1e-4)
        min_learning_rate = kwargs.get('min_learning_rate', 1e-8)
        decay_rate = kwargs.get('decay_rate', 1.0)
        clip_threshold = kwargs.get('clip_threshold', 5.0)
        dataset_size = kwargs.get('dataset_size', 7000)
        keep_prob = kwargs.get('keep_prob', 1.0)
        session = tf.get_default_session()
        if train:
            return session.run(
                [self.s, self.j, self.o, self.global_step],
                feed_dict={
                    self.w: tokens,
                    self.n: lengths,
                    self.s_: targets,
                    self.learning_rate: learning_rate,
                    self.min_learning_rate: min_learning_rate,
                    self.decay_rate: decay_rate,
                    self.clip_threshold: self.clip_threshold,
                    self.dataset_size: dataset_size,
                    self.keep_prob: keep_prob
                }
            )[:2]
        elif targets is not None:
            return session.run(
                [self.s, self.j],
                feed_dict={
                    self.w: tokens,
                    self.n: lengths,
                    self.s_: targets,
                    self.keep_prob: 1.0
                }
            )
        else:
            return session.run(
                [self.s],
                feed_dict={
                    self.w: tokens,
                    self.n: lengths,
                    self.keep_prob: 1.0
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


def load_encoder_dataset2(sentences, oracle=None):
    dataset = TFRecordDataset([path.join(DSSDIR, sentence+'.tfr')
                               for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'w': tf.FixedLenSequenceFeature([], tf.string,
                                                        allow_missing=True),
                        'n': tf.FixedLenFeature([], tf.int64)
                    }
                )
        )

    if oracle is None:
        return dataset.map(lambda feature: (feature['w'], feature['n']))
    else:
        indices = {s: n for n, s in enumerate(sentences)}
        return dataset.map(lambda feature: \
            (feature['w'],
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
