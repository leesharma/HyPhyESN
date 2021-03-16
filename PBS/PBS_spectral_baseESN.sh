#!/bin/bash
#
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=40:mpiprocs=40
#PBS -l place=scatter:excl
#PBS -N spectral_baseESN_NAME
#PBS -q standard
##PBS -r y
#PBS -j oe
#PBS -A ARLAP00581800
#
#
echo job ${JOBID} starting at `date` on `hostname`
echo starting in `pwd`
#
#
cd /p/home/mrziema/projects/HyPhyESN
julia train/spectral_baseESN.jl

set st=$status
echo "Program ended with status $st on `date`"

exit $st
