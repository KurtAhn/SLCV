#!/usr/bin/env python
from __init__ import load_config
from nltk.tokenize import word_tokenize
from os import path
import re
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-v', '--vocab', dest='vocab', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    with open(path.join(VOCDIR, a.vocab+'.voc')) as f:
        vocab = [l.rstrip() for l in f]

    for s in sentences:
        with open(path.join(TXTDIR, s+'.txt')) as f:
            line = next(f)
            if '*' in line:
                continue
            words = word_tokenize(line.lower())
        if all(w in vocab for w in words):
            print1(s)
