#========== settings ==========#
PROJECT_PATH=simvla_twin2
ROOT_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengdongzhou-240108390137
#========== !NOTE! ==========#
RUN_MODE=simvla_vae
use_predict_future_prop=False
batch_size=16
use_action_ts_head=True
use_one_embed=True
use_multi_scaling=False
mlp_type=ffn
decoder_num_blocks=6
robot_platform=16_li
proj_type=relu_linear
ffn_type=relu
expand_inner_ratio=1
linear_drop_ratio=0.0
multi_queries_num=1
multi_query_norm_type=layernorm
action_norm=layernorm
use_vae=True
MODE=${RUN_MODE}_inner${expand_inner_ratio}_proj_type_${proj_type}_ffn_type_${ffn_type}_mlp_${mlp_type}_decoder_num_blocks_${decoder_num_blocks}
#========== !NOTE! ==========#
use_l1_regression=True
num_images_in_input=1
wandb_entity=chenghaha
wandb_project=robotwin
wandb_log_freq=1
use_proprio=False
use_diffusion=False
use_film=False
num_steps_before_decay=20000
save_freq=40000
max_steps=40000
vla_path=$ROOT_PATH/ai_models/openvla/openvla-7b
data_root_dir=$ROOT_PATH/datasets/openvla/modified_libero_rlds
dataset_name=libero_4_task_suites_no_noops
run_root_dir=$ROOT_PATH/vla_projects/$PROJECT_PATH/results/$RUN_MODE
#========== get run_id ==========#
note_parts=("${MODE}")

# if [ "$use_l1_regression" = "True" ]; then
#     note_parts+=("L1_regression")
# fi

# if [ "$num_images_in_input" == 1 ]; then
#     note_parts+=("3rd_person_img")
# else
#     note_parts+=("3rd_person_img_and_wrist")
# fi

# if [ "$use_l1_regression" = "True" ]; then
#     note_parts+=("proprio_state")
# fi

# if [ "$use_film" = "True" ]; then
#     note_parts+=("Film")
# fi
note_parts+=("M$max_steps-F$save_freq-D$num_steps_before_decay")
run_id_note_value=$(IFS='--'; echo "${note_parts[*]}")

#========== enter environment ==========#
conda activate openvla-oft
cd $ROOT_PATH/vla_projects/$PROJECT_PATH
export PYTHONPATH=$ROOT_PATH/vla_projects/$PROJECT_PATH

#========== run ==========#
WANDB_CONSOLE=off WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 400000 \
  --max_steps 400005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "$data_name" \
  --run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
  > logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &