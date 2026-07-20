#!/bin/bash
#
#SBATCH --job-name=BAX_L_gradW_3
#SBATCH --output=out_BAX_L_gradW_3.log
#
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=node-98
#SBATCH --mail-type=FAIL
#SBATCH --time=168:00:00
# SBATCH --begin=now+10hours
# SBATCH --gres=gpu:3g.20gb:1

# Load module
export WANDB_MODE=offline

python -m graphphysics.train \
            --project_name=AX \
            --training_parameters_path=training_config/BAX_L_gradW.json \
            --num_epochs=20 \
            --init_lr=0.001 \
            --batch_size=1 \
            --warmup=500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=BAX_L_gradW_3 \
            --no_edge_feature \
            --use_previous_data=true \
            --previous_data_start=4 \
            --previous_data_end=7 \
            --seed=3 \
            # --model_path=checkpoints/AX_baselineCur_2.ckpt \
