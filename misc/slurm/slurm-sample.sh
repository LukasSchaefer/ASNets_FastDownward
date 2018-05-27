#!/bin/bash
## Usage:
## This scripts can ONLY call the sampling not the traversing block of
## ./fast-sample.py
## The environment variable TARGET_DIR is the directory to which to
## sample the data (replaces --target-folder). If no given, no target
## folder is specified. --target-file and --temporary-folder is NOT
## available.
## Provide arguments in the same way as for fast-sample.py EXCEPT:
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
module load Python/2.7.11-goolf-1.7.20

##SETUP VARIABLE
if [ ! -z ${TARGET_DIR+x} ]; then
	if [ ! -d "$TARGET_DIR" ]; then
		mkdir $TARGET_DIR
	fi
	if [ ! -d "$TARGET_DIR" ]; then
		(>&2 echo "error: Unable to make TARGET_DIR")
		exit 1
	fi
	
	PATH_DATA=$TMPDIR/data
	if [ ! -d "$PATH_DATA" ]; then
		mkdir $PATH_DATA
	fi
	if [ ! -d "$PATH_DATA" ]; then
		(>&2 echo "error: Unable to make DATA DIR")
		exit 3
	fi
fi

PATH_TMP=$TMPDIR/tmp
if [ ! -d "$PATH_TMP" ]; then
    mkdir $PATH_TMP
fi
if [ ! -d "$PATH_TMP" ]; then
	(>&2 echo "error: Unable to make TMP DIR")
    exit 2
fi

##RUN
if [ ! -z ${TARGET_DIR+x} ]; then
	if $NDOWNWARD/fast-sample.py -tmp $PATH_TMP -t $PATH_DATA "$@"; then
		mv $PATH_DATA/* $TARGET_DIR
		echo "finished"
	else
		(>&2 echo "error: Unable to sample data")
	fi
else
	if $NDOWNWARD/fast-sample.py -tmp $PATH_TMP "$@"; then
		echo "finished"
	else
		(>&2 echo "error: Unable to sample data")
	fi
fi

