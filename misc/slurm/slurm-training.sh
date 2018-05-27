#!/bin/bash

#SBATCH --job-name=fast_train
#SBATCH --time=04:00:00
#SBATCH --mem=3G

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20
source ~/bin/kerascpu/bin/activate


##RUN
if $NDOWNWARD/fast-training.py "$@"; then
	mv $PATH_DATA/* $TARGET_DIR
	echo "finished"
else
	(>&2 echo "error: Unable to train network(s)")
fi
deactivate
