from __future__ import print_function
import json
import sys
from os import path

def load_config(filepath):
    """
    Read in a config file and create lots of global variables.

    filepath: Where the config file is
    """
    global config
    # Under the "data" section
    global cfg_data
    # Under the "directories" section
    global cfg_dir
    # Under the "synthesizer" section
    global cfg_syn
    # Under the "encoder" section
    global cfg_enc
    # Under the "unpacker" section
    global cfg_unp
    # Under the "log" section
    global cfg_log

    # Project directory (one level above src/)
    global PRJDIR
    # Source directory (src/)
    global SRCDIR
    # Resource directory where HTS question file is
    global RESDIR
    # Data directory where pretty much everything is
    global DATDIR

    # The following are entries in the "directories" section
    # For XYZDIR, corresponding config file attribute is "xyz", eg:
    # "directories": {
    # ...
    #   "xyz": "path/to/xyz"
    # ...
    # }

    # Where the source waveforms are
    global WAVDIR
    # Where the transcripts are
    global TXTDIR
    # Where the HTS alignment files are
    global HTS1DIR
    # Kind of legacy (of past mistakes) where realigned HTS files are
    # Should be removed I guess
    global HTS2DIR
    # Where the initial binarized frame-level linguistic feature files are
    global LAB1DIR
    # Where the binarized linguistic features after trimming silence are
    global LAB2DIR
    # Where the final normalized linguistic features are
    global LAB3DIR
    # Where the mean and standard deviation of linguistic features are
    global LABSDIR
    # Where the binarized state-level features are
    global LABD1DIR
    # Where the silence-trimmed state-level features are
    global LABD2DIR
    # Where the normalized state-level features are
    global LABD3DIR
    # I forgot to add this... but there should be a folder with mean and
    # standard deviation of state-level features
    global LABDSDIR
    # Where the state durations are
    global DUR1DIR
    # Where the state durations after trimming silence
    global DUR2DIR
    # Where the duration mean and standard deviation are
    global DURSDIR
    # Where the MagPhase output files (.mag, .real, .imag, .lf0) are
    global ACO1DIR
    # Where the silence-trimmed MagPhase output files are
    global ACO2DIR
    # Where the (delta-appended) merged Magphase output files (as .aco) are
    global ACO3DIR
    # Where the acoustic mean and standard deviation are
    global ACOSDIR
    # Where the tokenized (using Stanford CoreNLP) transcripts are
    global TOKDIR
    # Where the word embedding data is
    global EMBDIR
    # Where the oracle values of Synthesizer models are stored
    global ORCSDIR
    # Where the oracle values of Encoder models are stored
    global ORCEDIR
    # Where the oracle values of Unpacker models are stored
    global ORCUDIR
    # Where the TFRecord files (.tfr) for training Synthesizer are
    global TFRSDIR
    # Where the TFRecord files (.tfr) for training Encoder are
    global TFREDIR
    # Where the TFRecord files (.tfr) for training Unpacker are
    global TFRUDIR
    # Where Synthesizer definitions are stored
    global MDLSDIR
    # Where Encoder definitions are stored
    global MDLEDIR
    # Where Unpacker definitions are stored
    global MDLUDIR
    # Where synthesized waveforms are stored
    global SYNDIR

    # Logging variables
    # Redirect stdout to
    global STDOUT
    # Redirect stderr to
    global STDERR
    # I don't think this is used but it's supposed to suppress certain outputs if set False
    global VERBOSE

    # Seed used for shuffling and splitting datasets
    # Not used for initializing neural nets
    global SEED

    with open(filepath) as f:
        config = json.load(f)
    cfg_data = config['data']
    cfg_dir = config['directories']
    cfg_syn = config.get('synthesizer', {})
    cfg_enc = config.get('encoder', {})
    cfg_unp = config.get('unpacker', {})
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
    LABSDIR = cfg_dir.get('labs', path.join(DATDIR, 'labs')).replace('$', PRJDIR)
    ACO1DIR = cfg_dir.get('aco1', path.join(DATDIR, 'aco1')).replace('$', PRJDIR)
    ACO2DIR = cfg_dir.get('aco2', path.join(DATDIR, 'aco2')).replace('$', PRJDIR)
    ACO3DIR = cfg_dir.get('aco3', path.join(DATDIR, 'aco3')).replace('$', PRJDIR)
    ACOSDIR = cfg_dir.get('acos', path.join(DATDIR, 'acos')).replace('$', PRJDIR)
    LABD1DIR = cfg_dir.get('labd1', path.join(DATDIR, 'labd1')).replace('$', PRJDIR)
    LABD2DIR = cfg_dir.get('labd2', path.join(DATDIR, 'labd2')).replace('$', PRJDIR)
    LABD3DIR = cfg_dir.get('labd3', path.join(DATDIR, 'labd3')).replace('$', PRJDIR)
    DUR1DIR = cfg_dir.get('dur1', path.join(DATDIR, 'dur1')).replace('$', PRJDIR)
    DUR2DIR = cfg_dir.get('dur2', path.join(DATDIR, 'dur2')).replace('$', PRJDIR)
    DURSDIR = cfg_dir.get('durs', path.join(DATDIR, 'durs')).replace('$', PRJDIR)
    TOKDIR = cfg_dir.get('tok', path.join(DATDIR, 'tok')).replace('$', PRJDIR)
    EMBDIR = cfg_dir.get('emb', path.join(DATDIR, 'emb')).replace('$', PRJDIR)
    ORCSDIR = cfg_dir.get('orcs', path.join(DATDIR, 'orcs')).replace('$', PRJDIR)
    ORCEDIR = cfg_dir.get('orce', path.join(DATDIR, 'orce')).replace('$', PRJDIR)
    ORCUDIR = cfg_dir.get('orcu', path.join(DATDIR, 'orcu')).replace('$', PRJDIR)
    TFRSDIR = cfg_dir.get('tfrs', path.join(DATDIR, 'tfrs')).replace('$', PRJDIR)
    TFREDIR = cfg_dir.get('tfre', path.join(DATDIR, 'tfre')).replace('$', PRJDIR)
    TFRUDIR = cfg_dir.get('tfru', path.join(DATDIR, 'tfru')).replace('$', PRJDIR)
    MDLSDIR = cfg_dir.get('mdls', path.join(DATDIR, 'mdls')).replace('$', PRJDIR)
    MDLEDIR = cfg_dir.get('mdle', path.join(DATDIR, 'mdle')).replace('$', PRJDIR)
    MDLUDIR = cfg_dir.get('mdlu', path.join(DATDIR, 'mdlu')).replace('$', PRJDIR)
    SYNDIR = cfg_dir.get('syn', path.join(DATDIR, 'syn')).replace('$', PRJDIR)

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
    """
    Print to stdout
    """
    print(*args, file=STDOUT, **kwargs)


def print2(*args, **kwargs):
    """
    Print to stderr
    """
    print(*args, file=STDERR, **kwargs)


def flush1():
    STDOUT.flush()


def flush2():
    STDERR.flush()
