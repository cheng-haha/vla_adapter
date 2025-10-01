PROJECT_PATH=VLA-Adapter
###
 # @Description: 
 # @Date: 2025-09-29 19:41:56
 # @LastEditTime: 2025-09-29 21:07:52
 # @FilePath: \vla_adapter\run_scripts\server_deploy\deploy_vla_port8886.sh
### 
export PYTHONPATH=/inspire/hdd/global_user/chengdongzhou-240108390137/vla_projects/$PROJECT_PATH
pretrained_checkpoint=/inspire/hdd/global_user/chengdongzhou-240108390137/ai_models/VLA-Adapter/LIBERO-Long-Pro
port=8885
model_family=openvla
device=0

python experiments/robot/server_deploy/deploy.py \
        --pretrained_checkpoint $pretrained_checkpoint \
        --model_family $model_family \
        --port $port \
        --device $device
