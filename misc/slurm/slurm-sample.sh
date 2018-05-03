#!/bin/bash
## Usage:
## This scripts can ONLY call the sampling not the traversing block of
## ./fast-sample.py
## The first argument is the folder where the sampled data shall be copied to.
## Then provide arguments in the same way as for fast-sample.py EXCEPT:
## Do NOT provide
##  --temporary-folder
##  --target-file
##  --target-folder
##    The script sets them itself and cannot use the option to store them in a
##    target file (as multiple runs can be done in parallel on different
##    compute nodes.


#SBATCH --job-name=fast_sample
#SBATCH --time=02:00:00
#SBATCH --mem=3G

##LOAD MODULES


TARGET_DIR=$1
if [ ! -d "$TARGET_DIR" ]; then
    mkdir $TARGET_DIR
fi
shift

PATH_TMP=$TMPDIR/tmp
if [ ! -d "$PATH_TMP" ]; then
    mkdir $PATH_TMP
fi


PATH_DATA=$TMPDIR/data
if [ ! -d "$PATH_DATA" ]; then
    mkdir $PATH_DATA
fi


if $NDOWNWARD/fast-sample.py -tmp $PATH_TMP -t $PATH_DATA "$@"; then
    mv $PATH_DATA/* $TARGET_DIR
fi


