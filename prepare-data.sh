#!/usr/bin/env bash

PRJDIR=`dirname $0`
SRCDIR=$PRJDIR/src
DATDIR=$PRJDIR/data
CFG=$PRJDIR/config/interpolate.json

source $PRJDIR/setup.sh
source activate thesis

# $SRCDIR/extract-acoustics.py -s $DATDIR/extract-acoustics.scp -c $CFG > \
#     $DATDIR/realign-states.scp || exit -1
# $SRCDIR/realign-states.py -s $DATDIR/realign-states.scp -c $CFG > \
#     $DATDIR/trim-acoustics.scp || exit -1
#$SRCDIR/create-labels.py -s $DATDIR/trim-acoustics.scp -c $CFG || exit -1
#$SRCDIR/trim-acoustics.py -s $DATDIR/trim-acoustics.scp -c $CFG || exit -1
$SRCDIR/delta-acoustics.py -s $DATDIR/trim-acoustics.scp -c $CFG > \
    $DATDIR/i-validate-transcripts.scp || exit -1
$SRCDIR/validate-transcripts.py -s $DATDIR/validate-transcripts.scp -c $CFG > \
    $DATDIR/i-create-pairset.scp || exit -1
$SRCDIR/compute-stats.py -s $DATDIR/i-create-pairset.scp -c $CFG || exit -1
$SRCDIR/create-pairset.py -s $DATDIR/i-create-pairset.scp -c $CFG > \
    $DATDIR/i-train.scp || exit -1

source deactivate
