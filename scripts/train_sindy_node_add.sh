#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --mem=8000
#SBATCH --job-name=train_sindy_node_add
#SBATCH --output=log/train_sindy_node_add-%j.out
#SBATCH --error=log/train_sindy_node_add-%j.err
#SBATCH --workdir=/home/simonwa/home/ode_test.jl/scripts

julia train_sindy_node_add.jl ${1}
