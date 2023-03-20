#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --mem=8000
#SBATCH --job-name=refine_sindy_node_cat
#SBATCH --output=log/refine_sindy_node_cat-%j.out
#SBATCH --error=log/refine_sindy_node_cat-%j.err
#SBATCH --workdir=/home/simonwa/home/SymbolicNODE.jl/scripts

julia refine_sindy_node_cat.jl $*