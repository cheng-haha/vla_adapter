#========== settings ==========#
PROJECT_PATH=SimVLA_Condition
ROOT_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengdongzhou-240108390137
#========== !NOTE! ==========#
RUN_MODE=simvla_q2a
use_predict_future_prop=False
batch_size=8
use_action_ts_head=True
use_one_embed=True
use_multi_scaling=False
mlp_type=moe
decoder_num_blocks=1
robot_platform=aloha
without_head_drop_out=True
proj_type=gelu_linear
ffn_type=gelu
num_experts=4
expand_inner_ratio=2
top_k=2
MODE=${RUN_MODE}_inner${expand_inner_ratio}_proj_type_${proj_type}_ffn_type_${ffn_type}_mlp_${mlp_type}_decoder_num_blocks_${decoder_num_blocks}_num_experts${num_experts}_top_k{$top_k}
#========== !NOTE! ==========#
use_l1_regression=True
num_images_in_input=1     
wandb_entity=chenghaha
wandb_project=fastvla
wandb_log_freq=1
use_proprio=True
use_diffusion=False
use_film=True
num_steps_before_decay=15000
save_freq=10000
max_steps=30000
vla_path=$ROOT_PATH/ai_models/openvla/openvla-7b
data_root_dir=$ROOT_PATH/vla_projects/robotwin_data/openvla_oft/tfds
dataset_name=aloha_dual_bottles_pick_hard_d435_20
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
WANDB_CONSOLE=off WANDB_MODE=offline python -m debugpy --listen 1234 --wait-for-client '/opt/conda/envs/openvla-oft/bin/torchrun' --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$vla_path" \
  --data_root_dir "$data_root_dir" \
  --dataset_name "$dataset_name" \
  --run_root_dir "$run_root_dir" \
  --use_l1_regression "$use_l1_regression" \
  --use_diffusion "$use_diffusion" \
  --use_film "$use_film" \
  --num_images_in_input "$num_images_in_input" \
  --use_proprio "$use_proprio" \
  --batch_size "$batch_size" \
  --learning_rate 5e-5 \
  --num_steps_before_decay "$num_steps_before_decay" \
  --max_steps "$max_steps" \
  --save_freq "$save_freq" \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "$wandb_entity" \
  --wandb_project "$wandb_project" \
  --wandb_log_freq "$wandb_log_freq" \
  --run_id_note "$run_id_note_value" \
  --use_predict_future_prop "$use_predict_future_prop" \
  --use_action_ts_head "$use_action_ts_head" \
  --use_one_embed "$use_one_embed" \
  --use_multi_scaling "$use_multi_scaling" \
  --mlp_type "$mlp_type" \
  --decoder_num_blocks "$decoder_num_blocks" \
  --robot_platform "$robot_platform" \
  --proj_type "$proj_type" \
  --ffn_type "$ffn_type" \
  --expand_inner_ratio "$expand_inner_ratio" \
  --num_experts "$num_experts" \
  --top_k "$top_k" 