#!/bin/bash

#SBATCH --job-name=lift_test_custom
#SBATCH --nodes=1
#SBATCH --tasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --account=AERO026062
#SBATCH --time=0:30:0
#SBATCH --mail-user=pz24779@bristol.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=veryshort
#SBATCH --mem=20G

##Loading modules:
module load openmpi/5.0.3

export SU2_RUN=/user/work/pz24779/custom_su2/bin
export SU2_HOME=/user/work/pz24779/custom_su2
export PATH=$PATH:$SU2_RUN
export PYTHONPATH=$PYTHONPATH:$SU2_RUN

##Run
mpirun -n 28 /user/work/pz24779/custom_su2/bin/SU2_CFD_AD adj_config.cfg

