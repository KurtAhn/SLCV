#!/usr/bin/env bash

PRJDIR=`dirname $0`
SRCDIR=$PRJDIR/src
DATDIR=$PRJDIR/data
CFG=$PRJDIR/config/test-config.json

source $PRJDIR/setup.sh
source activate thesis

# $SRCDIR/extract-acoustics.py -s $DATDIR/extract-acoustics.scp -c $CFG > \
#     $DATDIR/realign-states.scp || exit -1
# $SRCDIR/realign-states.py -s $DATDIR/realign-states.scp -c $CFG > \
#     $DATDIR/trim-acoustics.scp || exit -1
$SRCDIR/create-labels.py -s $DATDIR/trim-acoustics.scp -c $CFG || exit -1
$SRCDIR/trim-acoustics.py -s $DATDIR/trim-acoustics.scp -c $CFG || exit -1
$SRCDIR/delta-acoustics.py -s $DATDIR/trim-acoustics.scp -c $CFG > \
    $DATDIR/validate-transcripts.scp || exit -1
$SRCDIR/validate-transcripts.py -s $DATDIR/validate-transcripts.scp -c $CFG > \
    $DATDIR/create-pairset.scp || exit -1
$SRCDIR/compute-stats.py -s $DATDIR/create-pairset.scp -c $CFG || exit -1
$SRCDIR/create-pairset.py -s $DATDIR/create-pairset.scp -c $CFG > \
    $DATDIR/train.scp || exit -1

source deactivate
