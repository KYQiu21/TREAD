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
#SBATCH --time=00:30:00

module load cuda/12.1

conda activate /raven/u/kqiu/conda_env/tread

cd /raven/u/kqiu/tool/TREAD/
tread train train_example/repeatsdb/repeatsdb.fasta train_example/repeatsdb/repeatsdb_annotations.tsv --task multitask -o my_repeatsdb_model
tread predict predict_example/Q60773.fasta --weights my_repeatsdb_model/model.pt -o repeatsdb_custom_prediction
tread predict predict_example/multi.fasta --weights my_repeatsdb_model/model.pt -o repeatsdb_custom_prediction_multi