PROJECT_PATH=vla_adapter
###
 # @Description: 
 # @Date: 2025-09-29 21:30:22
 # @LastEditTime: 2025-09-29 21:30:51
 # @FilePath: \vla_adapter\run_scripts\eval\libero\libero_all2.sh
### 
export PYTHONPATH=/home/ubuntu/projects/vla_projects/$PROJECT_PATH
checkpoint=/home/ubuntu/projects/ai_models/iMihayo/vla_libero_one


# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --use_film False \
#   --pretrained_checkpoint /home/ubuntu/projects/ai_models/iMihayo/vla_libero_one \
#   --task_suite_name libero_10 \
#   --use_pro_version True

# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --use_film False \
#   --pretrained_checkpoint $checkpoint \
#   --task_suite_name libero_spatial \
#   --use_pro_version True 

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint $checkpoint \
  --task_suite_name libero_object \
  --use_pro_version True 

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint $checkpoint \
  --task_suite_name libero_goal \
  --use_pro_version True 