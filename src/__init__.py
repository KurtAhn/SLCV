from __future__ import print_function
import json
import sys
from os import path

def load_config(filepath):
    global config, cfg_data, cfg_dir, cfg_net, cfg_log,\
           PRJDIR, SRCDIR, RESDIR, DATDIR,\
           WAVDIR, TXTDIR, \
           HTS1DIR, HTS2DIR, \
           LAB1DIR, LAB2DIR, LAB3DIR,\
           ACO1DIR, ACO2DIR, ACO3DIR, \
           STTDIR, TOKDIR, EMBDIR, \
           ORCWDIR, ORCADIR, \
           DSATDIR, DSASDIR, DSSDIR, \
           MDLWDIR, MDLASDIR, MDLAEDIR, \
           SYNWDIR, SYNADIR, \
           STDOUT, STDERR, VERBOSE,\
           SEED

    with open(filepath) as f:
        config = json.load(f)
    cfg_data = config['data']
    cfg_dir = config['directories']
    cfg_net = config['network']
    cfg_log = config['log']

    PRJDIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
    SRCDIR = path.join(PRJDIR, 'src')
    RESDIR = path.join(PRJDIR, 'res')
    DATDIR = path.join(PRJDIR, 'data') # for default values
    WAVDIR = cfg_dir.get('wav', path.join(DATDIR, 'wav')).replace('$', PRJDIR)
    TXTDIR = cfg_dir.get('txt', path.join(DATDIR, 'txt')).replace('$', PRJDIR)
    HTS1DIR = cfg_dir.get('hts1', path.join(DATDIR, 'hts1')).replace('$', PRJDIR)
    HTS2DIR = cfg_dir.get('hts2', path.join(DATDIR, 'hts2')).replace('$', PRJDIR)
    LAB1DIR = cfg_dir.get('lab1', path.join(DATDIR, 'lab1')).replace('$', PRJDIR)
    LAB2DIR = cfg_dir.get('lab2', path.join(DATDIR, 'lab2')).replace('$', PRJDIR)
    LAB3DIR = cfg_dir.get('lab3', path.join(DATDIR, 'lab3')).replace('$', PRJDIR)
    ACO1DIR = cfg_dir.get('aco1', path.join(DATDIR, 'aco1')).replace('$', PRJDIR)
    ACO2DIR = cfg_dir.get('aco2', path.join(DATDIR, 'aco2')).replace('$', PRJDIR)
    ACO3DIR = cfg_dir.get('aco3', path.join(DATDIR, 'aco3')).replace('$', PRJDIR)
    STTDIR = cfg_dir.get('stt', path.join(DATDIR, 'stats')).replace('$', PRJDIR)
    TOKDIR = cfg_dir.get('tok', path.join(DATDIR, 'tok')).replace('$', PRJDIR)
    EMBDIR = cfg_dir.get('emb', path.join(DATDIR, 'emb')).replace('$', PRJDIR)
    ORCWDIR = cfg_dir.get('orcw', path.join(DATDIR, 'orcw')).replace('$', PRJDIR)
    ORCADIR = cfg_dir.get('orca', path.join(DATDIR, 'orca')).replace('$', PRJDIR)
    DSATDIR = cfg_dir.get('dsat', path.join(DATDIR, 'dsat')).replace('$', PRJDIR)
    DSASDIR = cfg_dir.get('dsas', path.join(DATDIR, 'dsas')).replace('$', PRJDIR)
    DSSDIR = cfg_dir.get('dss', path.join(DATDIR, 'dss')).replace('$', PRJDIR)
    MDLWDIR = cfg_dir.get('mdlw', path.join(DATDIR, 'mdlw')).replace('$', PRJDIR)
    MDLASDIR = cfg_dir.get('mdlas', path.join(DATDIR, 'mdlas')).replace('$', PRJDIR)
    MDLAEDIR = cfg_dir.get('mdlae', path.join(DATDIR, 'mdlae')).replace('$', PRJDIR)
    SYNWDIR = cfg_dir.get('synw', path.join(DATDIR, 'synw')).replace('$', PRJDIR)
    SYNADIR = cfg_dir.get('syna', path.join(DATDIR, 'syna')).replace('$', PRJDIR)

    STDOUT = cfg_log.get('stdout', '&1')
    STDOUT = sys.stdout if STDOUT == '&1' else \
             sys.stderr if STDOUT == '&2' else \
             open(STDOUT, 'w')

    STDERR = cfg_log.get('stderr', '&2')
    STDERR = sys.stdout if STDERR == '&1' else \
             sys.stderr if STDERR == '&2' else \
             open(STDERR, 'w')

    VERBOSE = cfg_log.get('verbose', True)

    SEED = 20180621

def print1(*args, **kwargs):
    print(*args, file=STDOUT, **kwargs)


def print2(*args, **kwargs):
    print(*args, file=STDERR, **kwargs)


def flush1():
    STDOUT.flush()


def flush2():
    STDERR.flush()
