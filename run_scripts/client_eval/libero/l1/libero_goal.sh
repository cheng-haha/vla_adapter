###
 # @Description: 
 # @Date: 2025-03-23 01:30:22
 # @LastEditTime: 2025-09-29 21:33:26
 # @FilePath: \vla_adapter\run_scripts\client_eval\libero\l1\libero_goal.sh
### 
#========= setting up environment =========#
PROJECT_PATH=vla_adapter
export PYTHONPATH=/home/ubuntu/projects/vla_projects/$PROJECT_PATH
vla_server_url=https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-1ac3d6e6-12d8-4935-bfd5-5a2483616812/user-df00c8a2-a471-4dd2-ba09-aab582ee29d2/vscode/ddd1d884-4989-4a2c-a4f1-c5247ea939ed/186c32d6-7fff-4955-b40f-aae20c2e7cd7/proxy/8080/

python eval/exps/libero_wtp/run_libero_metavla_eval_client.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --task_suite_name libero_goal \
  --use_pro_version True \
  --use_vla_server True \
  --vla_server_url $vla_server_url
