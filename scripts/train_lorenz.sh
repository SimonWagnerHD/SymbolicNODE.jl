#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=8000
#SBATCH --job-name=train_lorenz
#SBATCH --output=log/train_lorenz-%j.out
#SBATCH --error=log/train_lorenz-%j.err
#SBATCH --workdir=/home/simonwa/home/SymbolicNODE.jl/scripts

julia train_lorenz.jl ${1}