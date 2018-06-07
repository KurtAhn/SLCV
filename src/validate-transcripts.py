#!/usr/bin/env python
from __init__ import load_config
from nltk.tokenize import word_tokenize
import enchant
from os import path
import re
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *

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
