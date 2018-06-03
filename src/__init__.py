from __future__ import print_function
import json
import sys
from os import path


def load_config(path):
    with open(path) as f:
        global config = json.load(f)
    global cfg_data = config['data']
    global cfg_dir = config['directories']
    global cfg_net = config['network']
    global cfg_log = config['log']

    global PRJDIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
    global SRCDIR = path.join(PRJDIR, 'src')
    global RESDIR = path.join(PRJDIR, 'res')
    global DATDIR = path.join(PRJDIR, 'data') # for default values
    global MDLDIR = cfg_dir.get('models', path.join(DATDIR, 'models'))\
                    .replace('$', PRJDIR)
    global WAVDIR = cfg_dir.get('sounds', path.join(DATDIR, 'wav'))\
                    .replace('$', PRJDIR)
    global HTS1DIR = cfg_dir.get('raw-states', path.join(DATDIR, 'hts1'))\
                     .replace('$', PRJDIR)
    global HTS2DIR = cfg_dir.get('realigned-states', path.join(DATDIR, 'hts2'))\
                     .replace('$', PRJDIR)
    global LAB1DIR = cfg_dir.get('raw-linguistics', path.join(DATDIR, 'lab1'))\
                     .replace('$', PRJDIR)
    global LAB2DIR = cfg_dir.get('trimmed-linguistics', path.join(DATDIR, 'lab2'))\
                     .replace('$', PRJDIR)
    global LAB3DIR = cfg_dir.get('normalized-linguistics', path.join(DATDIR, 'lab3'))\
                     .replace('$', PRJDIR)
    global ACO1DIR = cfg_dir.get('raw-acoustics', path.join(DATDIR, 'aco1'))\
                     .replace('$', PRJDIR)
    global ACO2DIR = cfg_dir.get('trimmed-acoustics', path.join(DATDIR, 'aco2'))\
                     .replace('$', PRJDIR)
    global ACO3DIR = cfg_dir.get('delta-acoustics', path.join(DATDIR, 'aco3'))\
                     .replace('$', PRJDIR)
    global TXTDIR = cfg_dir.get('transcripts', path.join(DATDIR, 'txt'))\
                    .replace('$', PRJDIR)
    global STTDIR = cfg_dir.get('statistics', path.join(DATDIR, 'stats'))\
                    .replace('$', PRJDIR)
    global TRNDIR = cfg_dir.get('training', path.join(DATDIR, 'training'))\
                    .replace('$', PRJDIR)
    global VALDIR = cfg_dir.get('validation', path.join(DATDIR, 'validation'))\
                    .replace('$', PRJDIR)
    global TSTDIR = cfg_dir.get('testing', path.join(DATDIR, 'training'))\
                    .replace('$', PRJDIR)
    global SYNDIR = cfg_dir.get('synthesized', path.join(DATDIR, 'synthesized'))\
                    .replace('$', PRJDIR)

    global STDOUT = cfg_log.get('redirect-stdout', '&1')
    global STDOUT = sys.stdout if STDOUT == '&1' else \
                    sys.stderr if STDOUT == '&2' else \
                    open(STDOUT, 'w')
    global STDERR = cfg_log.get('redirect-stderr', '&2')
    global STDERR = sys.stdout if STDERR == '&1' else \
                    sys.stderr if STDERR == '&2' else \
                    open(STDERR, 'w')
    global VERBOSE = cfg_log.get('verbose', True)


def print1(*args, **kwargs):
    print(*args, file=STDOUT, **kwargs)


def print2(*args, **kwargs):
    print(*args, file=STDERR, **kwargs)


def flush1():
    STDOUT.flush()


def flush2():
    STDERR.flush()
