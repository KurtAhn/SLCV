#!/usr/bin/env python
from __init__ import *
from nltk.tokenize import word_tokenize
from argparse import ArgumentParser
import enchant
from os import path
import re


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    us = enchant.Dict('en_US')
    uk = enchant.Dict('en_GB')

    for s in sentences:
        with open(path.join(TXTDIR, s+'.txt')) as f:
            line = next(f)
            if '*' in line:
                continue
            words = re.sub(r'[^ A-Za-z\'*]','', line.lower()).split()
        if all(us.check(w.capitalize()) or uk.check(w.capitalize())
               for w in words):
            print(s)
