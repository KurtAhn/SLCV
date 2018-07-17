#!/usr/bin/env bash

PRJDIR=`dirname $0`
SRCDIR=$PRJDIR/src
DATDIR=$PRJDIR/data
WAVDIR=$DATDIR/wav
CFG=$PRJDIR/config/final.json
SCP=$DATDIR/both.scp

source $PRJDIR/setup.sh
source activate thesis

# echo -n "" > $SCP
# for f in `ls $WAVDIR/*.wav`; do
#     basename $f | cut -d. -f1 >> $SCP
# done

# echo Extracting acoustics >&2
# $SRCDIR/extract-acoustics.py -s $SCP -c $CFG || exit -1

# echo Realigning states >&2
# $SRCDIR/realign-states.py -s $SCP -c $CFG || exit -1

# echo Creating labels >&2
# $SRCDIR/create-labels.py -s $SCP -c $CFG || exit -1

# echo Trimming silent frames >&2
# $SRCDIR/trim-acoustics.py -s $SCP -c $CFG || exit -1

echo Merging acoustics >&2
$SRCDIR/merge-acoustics.py -s $SCP -c $CFG || exit -1

# echo Creating embedding >&2
# $SRCDIR/create-embedding.py -c $CFG
#
# echo Tokenizing transcripts >&2
# $SRCDIR/tokenize-transcripts.sh $SCP || exit -1
#
# echo Wordifying transcripts >&2
# $SRCDIR/wordify-transcripts.sh $SCP || exit -1
#
# echo Validating transcripts >&2
# $SRCDIR/validate-transcripts.py -s $SCP || exit -1

echo Computing statistics >&2
$SRCDIR/compute-stats.py -s $SCP -c $CFG || exit -1

echo Creating dataset >&2
$SRCDIR/create-trainset.py -s $SCP -c $CFG || exit -1

source deactivate
