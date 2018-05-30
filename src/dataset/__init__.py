import tensorflow as tf
import acoustic as ax
import struct
import numpy as np


AX_LEN = ax.AX_LEN * 3 + 1
LX_LEN = 609
WORD = 582

def count_examples(filepaths):
    n = 0
    for f in filepaths:
        for r in tf.python_io.tf_record_iterator(f):
            n += 1
    return n


def load_trainset(filepaths, unit='s'):
    st2d = tf.sparse_tensor_to_dense
    return tf.data.TFRecordDataset(filepaths)\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        unit: tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([LX_LEN], tf.float32),
                        'a': tf.FixedLenFeature([AX_LEN], tf.float32)
                    }
                )
        )\
        .map(
            lambda features: \
                (features['l'], features[unit], features['a'])
                # (tf.reshape(st2d(features['l'], [LX_LEN])),
                #  features[unit],
                #  tf.reshape(st2d(features['a'], [AX_LEN]))
                # )
        )


def load_synthset(filepaths):
    st2d = tf.sparse_tensor_to_dense
    return tf.data.TFRecordDataset(filepaths)\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        'l': tf.FixedLenFeature([1,LX_LEN])
                    }
                )
        )\
        .map(
            lambda features: \
                tf.reshape(st2d(features['a'], [LX_LEN]))
        )


# def read_lab_binary(path, chunk_size=8192):
#     with open(path, 'rb') as f:
#         x = []
#         while True:
#             chunk = f.read(chunk_size)
#             if chunk:
#                 x.extend([struct.unpack('f', chunk[n:n+4])[0]
#                           for n in range(0,len(chunk),4)])
#             else:
#                 break
#     return np.reshape(x, (-1, 609))


# def ling2vfr(ling, dur, t1, t2):
#     f1 = int(t1 // 625)
#     f2 = int(t2 // 625)
#
#     vfr = []
#     f = 0
#     i = 0
#
#     while f < f2:
#         if f1 <= f:
#             vfr.append(ling[int((f - f1) // 80)])
#         else:
#             n1 = i
#         f += dur[i]
#         i += 1
#         n2 = i
#
#     return np.vstack(vfr), n1, n2
#
#
# def words2vfr(words, dur, t1, t2):
#     f1 = int(t1 // 625)
#     f2 = int(t2 // 625)
#
#     vfr = []
#     f = 0
#     i = 0
#
#     while f < f2:
#         if f1 <= f:
#             print(int((f - f1) // 80))
#             vfr.append(words[int((f - f1) // 80)])
#         else:
#             n1 = i
#         f += dur[i]
#         i += 1
#         n2 = i
#
#     return np.vstack(words), n1, n2
