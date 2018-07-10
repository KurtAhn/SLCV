#!/usr/bin/env python2
"""
Modified version of extract_features_for_merlin.py.
Main differences:
    - --debug flag
    - Use senlst instead of converting all audio files in directory
    - Print successfully converted sentences
"""
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import magphase as mp
import libutils as lu
import libaudio as la
from os import path
from argparse import ArgumentParser


def extract(sentence, wavdir, outdir):
    try:
        mp.analysis_for_acoustic_modelling(path.join(wavdir, sentence+'.wav'),
                                           outdir,
                                           mag_dim=cfg_data.get('nm', 60),
                                           phase_dim=cfg_data.get('np', 45))
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print2('Error while extracting features from',
               path.join(wavdir, sentence+'.wav'),
               'to',
               outdir)
        flush2()
    else:
        print1(sentence)
        flush1()


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-d', '--debug', dest='debug', action='store_true')
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    if a.debug:
        for s in sentences:
            extract(s, WAVDIR, ACO1DIR)
    else:
        lu.run_multithreaded(extract, sentences, WAVDIR, ACO1DIR)
