#========== settings ==========#
PROJECT_PATH=simvla_twin2
ROOT_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengdongzhou-240108390137
#========== !NOTE! ==========#
RUN_MODE=simvla_v2a
use_predict_future_prop=False
batch_size=4
use_action_ts_head=True
use_one_embed=True
use_multi_scaling=False
mlp_type=ffn
decoder_num_blocks=4
robot_platform=aloha
proj_type=onlynorm
ffn_type=swiglu
expand_inner_ratio=1
linear_drop_ratio=0.0
multi_queries_num=25
multi_query_norm_type=layernorm
action_norm=layernorm
use_v2a=True
use_action_projector=True
MODE=${RUN_MODE}_inner${expand_inner_ratio}_proj_type_${proj_type}_ffn_type_${ffn_type}_mlp_${mlp_type}_decoder_num_blocks_${decoder_num_blocks}
#========== !NOTE! ==========#
use_l1_regression=True
num_images_in_input=1
wandb_entity=chenghaha
wandb_project=robotwin
wandb_log_freq=1
use_proprio=True
use_diffusion=False
use_film=True
num_steps_before_decay=1000
save_freq=2000
max_steps=2000
vla_path=$ROOT_PATH/ai_models/openvla/openvla-7b
data_root_dir=$ROOT_PATH/datasets/TianxingChen/RoboTwin2.0/tfds
dataset_name=handover_mic_aloha_agilex_50
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
WANDB_CONSOLE=off WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_v2a.py \
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
  --learning_rate 1e-4 \
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
  --linear_drop_ratio "$linear_drop_ratio" \
  --multi_query_norm_type "$multi_query_norm_type" \
  --multi_queries_num "$multi_queries_num" \
  --action_norm "$action_norm" \
  --use_v2a "$use_v2a" \
  --use_action_projector "$use_action_projector"
