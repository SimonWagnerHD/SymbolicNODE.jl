#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=8000
#SBATCH --job-name=refine_node
#SBATCH --output=log/refine_node-%j.out
#SBATCH --error=log/refine_node-%j.err
#SBATCH --workdir=/home/simonwa/home/SymbolicNODE.jl/scripts

julia refine_node.jl $*