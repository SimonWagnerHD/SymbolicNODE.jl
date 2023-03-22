#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --mem=8000
#SBATCH --job-name=train_sindy_node_cat
#SBATCH --output=log/train_sindy_node_cat-%j.out
#SBATCH --error=log/train_sindy_node_cat-%j.err
#SBATCH --workdir=/home/simonwa/home/SymbolicNODE.jl/scripts

julia train_sindy_node_cat.jl ${1}
