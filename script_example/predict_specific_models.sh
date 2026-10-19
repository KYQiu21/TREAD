#!/bin/bash -l

# Standard output and error:
#SBATCH -o /raven/u/kqiu/script_out/job.out.%j
#SBATCH -e /raven/u/kqiu/script_out/job.err.%j
# Initial working directory:
#SBATCH -D ./

# Job name
#SBATCH -J tread

#SBATCH --ntasks=1
#SBATCH --constraint="tread"

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000

#SBATCH --mail-type=none
#SBATCH --mail-user=kaiyu.qiu@tuebingen.mpg.de
#SBATCH --time=00:05:00

module load cuda/12.1

conda activate /raven/u/kqiu/conda_env/tread

cd /raven/u/kqiu/tool/TREAD/
# Six RepeatsDB folds annotation
tread predict predict_example/O95834.fasta -o repeatsdb_model --model repeatsdb

# Beta-propeller blade annotation
tread predict predict_example/O95834.fasta -o propeller_blade_model --model propeller-blade