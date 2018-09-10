#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MERLIN'])
from utils.compute_distortion import IndividualDistortionComp
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
import numpy as np
np.warnings.filterwarnings('ignore')
from argparse import ArgumentParser

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def c


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-d', '--directory', dest='directory', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    for sentence in sentences:


    # mcd_computer = IndividualDistortionComp()
    # distortions = {
    #     f: mcd_computer.compute_distortion(sentences, ACO2DIR, a.directory, '.'+f, d)
    #     for f, d in [('mag', ax.MAG_DIM),
    #                  ('real', ax.REAL_DIM),
    #                  ('imag', ax.IMAG_DIM),
    #                  ('lf0', ax.LF0_DIM)]
    # }
    #
    # print1('MAG:{:.3f}dB; '
    #        'REAL:{:.3f}dB; '
    #        'IMAG:{:.3f}dB; '
    #        'LF0 RMSE: {:.3f}Hz; CORR: {:.3f}; VUV: {:.3f}%'.format(
    #            distortions['mag'],
    #            distortions['real'],
    #            distortions['imag'],
    #            *distortions['lf0'][:2],
    #            distortions['lf0'][-1] * 100.0
    #        ))
