from __init__ import *
import tensorflow as tf
try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    from tensorflow.contrib.data import TFRecordDataset
import acoustic as ax
import numpy as np

NC = cfg_data.get('nc', 2)
EXP_F0 = cfg_data.get('exp', True)
USE_DELTA = cfg_data.get('delta', True)
CONST_RATE = cfg_data.get('const', True)
AX_DIM = ax.AX_DIM * 3 + 1 if USE_DELTA else ax.AX_DIM + 1
LX_DIM = cfg_data.get('linguistic-dim', 609)
WIP = cfg_data.get('word-in-phrase-index', 582)
PIS = cfg_data.get('phrase-in-sentence-index', 593)


def count_examples(filepaths):
    n = 0
    for f in filepaths:
        for r in tf.python_io.tf_record_iterator(f):
            n += 1
    return n


def load_trainset(filepaths):
    st2d = tf.sparse_tensor_to_dense
    return TFRecordDataset(filepaths)\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'w': tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([LX_DIM], tf.float32),
                        'a': tf.FixedLenFeature([AX_DIM], tf.float32)
                    }
                )
        )


def load_synthset(filepaths):
    st2d = tf.sparse_tensor_to_dense
    return TFRecordDataset(filepaths)\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        'l': tf.FixedLenFeature([1,LX_DIM])
                    }
                )
        )
