PROJECT_PATH=VLA-Adapter
###
 # @Description: 
 # @Date: 2025-09-29 20:51:29
 # @LastEditTime: 2025-09-29 21:07:59
 # @FilePath: \vla_adapter\run_scripts\server_deploy\deploy_vla_port8889.sh
### 
export PYTHONPATH=/inspire/hdd/global_user/chengdongzhou-240108390137/vla_projects/$PROJECT_PATH
pretrained_checkpoint=/inspire/hdd/global_user/chengdongzhou-240108390137/vla_projects/VLA-Adapter/outputs/libero_4_task_suites_no_noops/img2_miniTrue_propTrue_proTrue_filmFalse/configs+libero_4_task_suites_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--vla--20250926_201005--135000_chkpt
port=8889
model_family=openvla
device=3

python experiments/robot/server_deploy/deploy.py \
        --pretrained_checkpoint $pretrained_checkpoint \
        --model_family $model_family \
        --port $port \
        --device $device
