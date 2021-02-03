#!/bin/bash
#
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=40:mpiprocs=40
#PBS -l place=scatter:excl
#PBS -N spectral_baseESN_job4
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
julia train/spectral_base_job4.jl

set st=$status
echo "Program ended with status $st on `date`" 

exit $st
