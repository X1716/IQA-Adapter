#!/bin/bash

#SBATCH --job-name=train_IQA-Adapter
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=5                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8               # number of GPUs per node
#SBATCH --cpus-per-task=128         # number of cores per tasks
#SBATCH --time=0-4:59:59            # maximum execution time (HH:MM:SS)
#SBATCH --exclude=cn26,cn36


#####
## This is a SLURM script used to train IQA-Adapter. 
## Training parameters here assume training on 5 nodes with 8 80GB GPUs each and 128 virtual CPU cores
#####

######################
### Set enviroment ###
######################
#source activateEnvironment.sh
export GPUS_PER_NODE=8
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    "
export TRAIN_DATASET="CC3M"
export QUALITY_DATA_PATH="/path/to/feather/file/with/metric/values"
export L=0
export POS_ENC=0
export PROJ_TYPE="linear"
export OUT_DIR="/where/to/save/checkpoints" 
export SAVE_FREQ=1000
export FEATS="topiq_nr laion_aes" # whitespace separated list of names of IQA metrics used during training
export NUM_EPOCHS=3
export RESOLUTION=1024
export BSIZE=16
export NUM_TOKENS=2
export LR="1e-5"
export ADDITIONAL_ARGS="--pretrained_ip_adapter_path /path/to/model/trained on CC3M"
#export ADDITIONAL_ARGS="" # if training is performed on CC3M
export CONTAINER_PATH="/path/to/PyXis/container"
export MAIN_FOLDER_PATH="/path/to/your/workdir/on/frontend/server"
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER train_iqa_adapter.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0  --logging_dir ./logs --output_dir ${OUT_DIR} --resolution=${RESOLUTION} --train_batch_size=${BSIZE} --dataloader_num_workers=4 --save_steps=${SAVE_FREQ} --num_train_epochs=${NUM_EPOCHS} --use_flag 0 --normalize 1 --features ${FEATS} --pos_encode ${POS_ENC} --pos_encode_L ${L} ${ADDITIONAL_ARGS} --num_tokens ${NUM_TOKENS} --proj_type ${PROJ_TYPE} --learning_rate ${LR} --train_dataset ${TRAIN_DATASET} --quality_data_path ${QUALITY_DATA_PATH}" 
srun --container-image "${CONTAINER_PATH}" --container-mounts "${MAIN_FOLDER_PATH}":/test \
    bash -c "cd /workdir && cp -r /test/iqa_adapter ./ && cp /test/tutorial_train_LAION-SBS_sdxl_test.py ./ && pip install datasets && ${CMD}"