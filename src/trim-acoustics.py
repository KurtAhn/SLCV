#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MERLIN'])
from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover, trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from os import path
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    for x, n in {'mag': ax.MAG_DIM,
                 'real': ax.REAL_DIM,
                 'imag': ax.IMAG_DIM,
                 'lf0': ax.LF0_DIM}.items():
        remover = SilenceRemover(n_cmp=n,
                                 silence_pattern=['*-#+*'],
                                 label_type='hts')
        remover.remove_silence([path.join(ACO1DIR, s+'.'+x) for s in sentences],
                               [path.join(HTS2DIR, s+'.lab') for s in sentences],
                               [path.join(ACO2DIR, s+'.'+x) for s in sentences])
