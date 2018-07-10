from __init__ import *
from time import time
from collections import deque


class Report:
    def __init__(self, epoch, mode='t'):
        self._epoch = epoch
        self._mode = mode
        self._total_loss = 0.0
        self._iterations = 0
        self._durations = deque(maxlen=10)
        self._time = time()

    def report(self, loss):
        self._total_loss += loss
        self._iterations += 1

        t1 = time()
        self._durations.append(t1 - self._time)
        self._time = t1

        if VERBOSE:
            print2('\r\x1b[0;{m};40m'
                   'Epoch: {e}'
                   ' Iteration: {i}'
                   ' Loss: {l:.3e}'
                   ' Avg: {a:.3e}'
                   ' It./s: {s:.3f}'
                   '\x1b[0m'.format(
                m=37 if self._mode == 't' else 33,
                e=self._epoch,
                i=self._iterations,
                l=loss,
                a=self._total_loss / self._iterations,
                s=len(self._durations) / sum(self._durations)
            ), end='')

    @property
    def avg_loss(self):
        try:
            return self._total_loss / self._iterations
        except ZeroDivisionError:
            return 0

    @property
    def iterations(self):
        return self._iterations
