from __init__ import *
import tensorflow as tf
try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    from tensorflow.contrib.data import TFRecordDataset
import acoustic as ax
# import struct
import numpy as np


AX_DIM = ax.AX_DIM * 3 + 1
LX_DIM = cfg_data.get('linguistic-dim', 609)
WORD = cfg_data.get('word-position-index', 582)


# def count_examples(filepaths):
#     n = 0
#     for f in filepaths:
#         for r in tf.python_io.tf_record_iterator(f):
#             n += 1
#     return n


def load_trainset(filepaths, unit='s'):
    st2d = tf.sparse_tensor_to_dense
    return TFRecordDataset(filepaths)\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        unit: tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([LX_DIM], tf.float32),
                        'a': tf.FixedLenFeature([AX_DIM], tf.float32)
                    }
                )
        )\
        .map(
            lambda features: (features['l'], features[unit], features['a'])
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
        )\
        .map(
            lambda features: \
                tf.reshape(st2d(features['a'], [LX_DIM]))
        )
