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
#SBATCH --time=00:20:00

module load cuda/12.1

conda activate /raven/u/kqiu/conda_env/tread

cd /raven/u/kqiu/tool/TREAD/
tread train train_example/propeller/propeller.fasta train_example/propeller/propeller_annotations.tsv -o my_propeller_model