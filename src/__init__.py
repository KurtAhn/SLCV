from __future__ import print_function
import json


with open('config.json') as f:
    config = json.load(f)
cfg_data = config['data']
cfg_dir = config['directories']
cfg_net = config['network']
cfg_log = config['log']


import sys
from os import path


PRJDIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
SRCDIR = path.join(PRJDIR, 'src')
RESDIR = path.join(PRJDIR, 'res')
DATDIR = path.join(PRJDIR, 'data') # for default values
MDLDIR = cfg_dir.get('models', path.join(DATDIR, 'models'))\
         .replace('$', PRJDIR)
WAVDIR = cfg_dir.get('sounds', path.join(DATDIR, 'wav'))\
         .replace('$', PRJDIR)
HTS1DIR = cfg_dir.get('raw-states', path.join(DATDIR, 'hts1'))\
          .replace('$', PRJDIR)
HTS2DIR = cfg_dir.get('realigned-states', path.join(DATDIR, 'hts2'))\
          .replace('$', PRJDIR)
LAB1DIR = cfg_dir.get('raw-linguistics', path.join(DATDIR, 'lab1'))\
          .replace('$', PRJDIR)
LAB2DIR = cfg_dir.get('trimmed-linguistics', path.join(DATDIR, 'lab2'))\
          .replace('$', PRJDIR)
LAB3DIR = cfg_dir.get('normalized-linguistics', path.join(DATDIR, 'lab3'))\
          .replace('$', PRJDIR)
ACO1DIR = cfg_dir.get('raw-acoustics', path.join(DATDIR, 'aco1'))\
          .replace('$', PRJDIR)
ACO2DIR = cfg_dir.get('trimmed-acoustics', path.join(DATDIR, 'aco2'))\
          .replace('$', PRJDIR)
ACO3DIR = cfg_dir.get('delta-acoustics', path.join(DATDIR, 'aco3'))\
          .replace('$', PRJDIR)
TXTDIR = cfg_dir.get('transcripts', path.join(DATDIR, 'txt'))\
          .replace('$', PRJDIR)
STTDIR = cfg_dir.get('statistics', path.join(DATDIR, 'stats'))\
          .replace('$', PRJDIR)
TRNDIR = cfg_dir.get('training', path.join(DATDIR, 'training'))\
          .replace('$', PRJDIR)
VALDIR = cfg_dir.get('validation', path.join(DATDIR, 'validation'))\
          .replace('$', PRJDIR)
TSTDIR = cfg_dir.get('testing', path.join(DATDIR, 'training'))\
          .replace('$', PRJDIR)
SYNDIR = cfg_dir.get('synthesized', path.join(DATDIR, 'synthesized'))\
          .replace('$', PRJDIR)

STDOUT = cfg_log.get('redirect-stdout', '&1')
STDOUT = sys.stdout if STDOUT == '&1' else \
         sys.stderr if STDOUT == '&2' else \
         open(STDOUT, 'w')

STDERR = cfg_log.get('redirect-stderr', '&2')
STDERR = sys.stdout if STDERR == '&1' else \
         sys.stderr if STDERR == '&2' else \
         open(STDERR, 'w')

VERBOSE = cfg_log.get('verbose', True)

def print1(*args, **kwargs):
    print(*args, file=STDOUT, **kwargs)


def print2(*args, **kwargs):
    print(*args, file=STDERR, **kwargs)


def flush1():
    STDOUT.flush()


def flush2():
    STDERR.flush()
