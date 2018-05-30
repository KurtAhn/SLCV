#!/usr/bin/env bash

PRJDIR=`dirname $0`
SRCDIR=$PRJDIR/src
DATDIR=$PRJDIR/data
WAVDIR=$DATDIR/wav
LABDIR=$DATDIR/lab
ACODIR=$DATDIR/aco
ALNDIR=$DATDIR/hts
ALNRDIR=$DATDIR/hts2
TRNDIR=$DATDIR/train
TXTDIR=$DATDIR/txt

#magphase $WAVDIR $ACODIR 48000 || exit -1
#realign $DATDIR/realign.scp $ALNDIR $ACODIR 48000 $ALNRDIR || exit -1
#grep -F -x -v -f $DATDIR/crash.scp $DATDIR/realign.scp > $DATDIR/binarize.scp || exit -1
#$SRCDIR/binarize-lab.py -s $DATDIR/binarize.scp || exit -1
#grep -F -x -v -f `ls $DATDIR/crash*` realign.scp
# echo Trimming Acoustics
# $SRCDIR/trim-aco.py -s $DATDIR/binarize.scp > $DATDIR/trim-reject.scp || exit -1
# grep -F -x -v -f $DATDIR/trim-reject.scp $DATDIR/binarize.scp > $DATDIR/delta.scp || exit -1
# echo Computing Acoustics Delta
#$SRCDIR/delta-aco.py -s $DATDIR/delta.scp || exit -1
# echo Validating Transcripts
# $SRCDIR/validate-txt.py -s $DATDIR/delta.scp -t $TXTDIR > $DATDIR/tfrize.scp|| exit -1
#$SRCDIR/compute-stats.py -s $DATDIR/train.scp || exit -1
$SRCDIR/create-trainset.py -s $DATDIR/tfrize.scp > $DATDIR/train.scp || exit -1
