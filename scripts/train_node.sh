#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --mem=8000
#SBATCH --job-name=train_node
#SBATCH --output=log/train_node-%j.out
#SBATCH --error=log/train_node-%j.err
#SBATCH --workdir=/home/simonwa/home/ode_test.jl/scripts

julia train_node.jl ${1}