#!/bin/bash
#
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=40:mpiprocs=1:bigmem=1
#PBS -l place=scatter:excl
#PBS -N CFSR_baseESN_job1
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
julia train/CFSR_baseESN.jl

set st=$status
echo "Program ended with status $st on `date`"

exit $st
