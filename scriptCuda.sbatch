#!/bin/bash
#SBATCH --job-name=version1.1   # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rhn_a_farimani@yahoo.com     # Where to send mail	
#SBATCH --nodes=1                    # Run on a single CPU
#SBATCH --ntasks=1
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --account=p70971
#SBATCH --qos p70971_a40dual
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=version1.1_%j.log   # Standard output and error log
pwd; hostname; date
module load cuda/11.5.0-gcc-9.4.0-57cwc5i 
module load gcc/10.2.0-gcc-9.1.0-2aa5hfe
nvcc -std=c++11 -G -gencode arch=compute_80,code=sm_80 -lcurand  main.cu -o main.run


pwd; hostname; date

