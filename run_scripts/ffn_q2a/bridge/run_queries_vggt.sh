#========== settings ==========#
PROJECT_PATH=simvla_twin2
ROOT_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengdongzhou-240108390137
#========== !NOTE! ==========#
RUN_MODE=simvla_vggt_5
use_predict_future_prop=False
batch_size=12
use_action_ts_head=True
use_one_embed=True
use_multi_scaling=False
mlp_type=ffn
decoder_num_blocks=6
robot_platform=bridge
proj_type=gelu_linear
ffn_type=gelu
expand_inner_ratio=1
linear_drop_ratio=0.0
multi_queries_num=5
multi_query_norm_type=layernorm
action_norm=layernorm
use_fredf=False
use_3d_visual_regression=True
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
save_freq=30000
max_steps=30000
vla_path=$ROOT_PATH/ai_models/openvla/openvla-7b
data_root_dir=$ROOT_PATH/datasets/openx/data/origin
dataset_name=bridge
run_root_dir=$ROOT_PATH/vla_projects/$PROJECT_PATH/results/$RUN_MODE
vggt_model_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengdongzhou-240108390137/ai_models/facebook/VGGT-1B/model.pt
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
WANDB_CONSOLE=off WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_3d.py \
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
  --learning_rate 1e-5 \
  --num_steps_before_decay "$num_steps_before_decay" \
  --max_steps "$max_steps" \
  --save_freq "$save_freq" \
  --save_latest_checkpoint_only False \
  --image_aug False \
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
  --use_fredf "$use_fredf" \
  --vggt_model_path "$vggt_model_path" \
  --use_3d_visual_regression "$use_3d_visual_regression"
