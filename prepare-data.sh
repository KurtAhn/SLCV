#!/usr/bin/env bash

PRJDIR=`dirname $0`
SRCDIR=$PRJDIR/src
DATDIR=$PRJDIR/data/test

source $PRJDIR/setup.sh
rm $PRJDIR/config.json
ln -s $PRJDIR/config/test-config.json $PRJDIR/config.json
source activate thesis

# $SRCDIR/extract-acoustics.py -s $DATDIR/extract-acoustics.scp > \
#     $DATDIR/realign-states.scp || exit -1
# $SRCDIR/realign-states.py -s $DATDIR/realign-states.scp > \
#     $DATDIR/trim-acoustics.scp || exit -1
$SRCDIR/create-labels.py -s $DATDIR/trim-acoustics.scp || exit -1
$SRCDIR/trim-acoustics.py -s $DATDIR/trim-acoustics.scp || exit -1
$SRCDIR/delta-acoustics.py -s $DATDIR/trim-acoustics.scp > \
    $DATDIR/validate-transcripts.scp || exit -1
$SRCDIR/validate-transcripts.py -s $DATDIR/validate-transcripts.scp > \
    $DATDIR/create-trainset.scp || exit -1
$SRCDIR/compute-stats.py -s $DATDIR/create-trainset.scp || exit -1
$SRCDIR/create-trainset.py -s $DATDIR/create-trainset.scp > \
    $DATDIR/train.scp || exit -1

source deactivate
