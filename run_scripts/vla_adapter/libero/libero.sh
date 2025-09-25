#!/bin/bash
###
 # @Description: 
 # @Date: 2025-09-25 22:13:40
 # @LastEditTime: 2025-09-25 22:39:08
 # @FilePath: \vla_adapter\run_scripts\vla_adapter\libero\libero.sh
### 

#========== Basic Settings ==========#
PROJECT_PATH=VLA-Adapter
ROOT_PATH=/inspire/hdd/global_user/chengdongzhou-240108390137
export WANDB_CONSOLE=off
export WANDB_MODE=offline
export PYTHONPATH=$ROOT_PATH/vla_projects/$PROJECT_PATH
#========== Training Configuration ==========#
# Dataset and paths
data_name=libero_4_task_suites_no_noops
data_root_dir=data/libero
vlm_path=$ROOT_PATH/ai_models/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b
llm_local_path=$ROOT_PATH/ai_models/Qwen/Qwen2.5-0.5B
config_file_path=pretrained_models/configs

# Training parameters
batch_size=16
grad_accumulation_steps=1
learning_rate=2e-4
max_steps=150005
num_steps_before_decay=150000
save_freq=5000

# Model configuration
num_images_in_input=2
lora_rank=64
use_film=False
use_proprio=True
use_lora=True
use_fz=False
use_minivlm=True
image_aug=True
save_latest_checkpoint_only=False
merge_lora_during_training=True
use_pro_version=True

# Wandb settings
wandb_entity=chenghaha
wandb_project=vla_adapter

# Generate timestamp and run ID
current_time=$(date +"%Y%m%d_%H%M%S")
run_id_note="vla--$current_time"

# Build MODE string with important configuration variables (excluding those already in run_id)
# run_id already includes: config_file_path, dataset_name, batch_size*grad_accumulation_steps, learning_rate, lora_rank, image_aug
MODE="img${num_images_in_input}_mini${use_minivlm}_prop${use_proprio}_pro${use_pro_version}_film${use_film}"

# Build run_root_dir using MODE
run_root_dir="outputs/${data_name}/${MODE}/${current_time}"


mkdir -p logs

#========== Training Execution ==========#
python -m debugpy --listen 1234 --wait-for-client '/root/anaconda3/envs/vla-adapter/bin/torchrun' --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path $vlm_path \
  --llm_local_path $llm_local_path \
  --config_file_path $config_file_path \
  --data_root_dir $data_root_dir \
  --dataset_name $data_name \
  --run_root_dir $run_root_dir \
  --use_film $use_film \
  --num_images_in_input $num_images_in_input \
  --use_proprio $use_proprio \
  --use_lora $use_lora \
  --use_fz $use_fz \
  --use_minivlm $use_minivlm \
  --image_aug $image_aug \
  --num_steps_before_decay $num_steps_before_decay \
  --max_steps $max_steps \
  --save_freq $save_freq \
  --save_latest_checkpoint_only $save_latest_checkpoint_only \
  --merge_lora_during_training $merge_lora_during_training \
  --batch_size $batch_size \
  --grad_accumulation_steps $grad_accumulation_steps \
  --learning_rate $learning_rate \
  --lora_rank $lora_rank \
  --use_pro_version $use_pro_version \
  --wandb_entity "$wandb_entity" \
  --wandb_project "$wandb_project" \
  --run_id_note $run_id_note 

echo "Training started with run ID: $run_id_note"
echo "Output directory: $run_root_dir"
echo "Log file: logs/$run_id_note.log"
echo "Process ID: $!"