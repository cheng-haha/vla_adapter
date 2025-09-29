PROJECT_PATH=VLA-Adapter
###
 # @Description: 
 # @Date: 2025-09-29 19:41:56
 # @LastEditTime: 2025-09-29 21:07:52
 # @FilePath: \vla_adapter\run_scripts\server_deploy\deploy_vla_port8886.sh
### 
export PYTHONPATH=/inspire/hdd/global_user/chengdongzhou-240108390137/vla_projects/$PROJECT_PATH
pretrained_checkpoint=/inspire/hdd/global_user/chengdongzhou-240108390137/vla_projects/nora/outputs/da_libero_4_task_suites_no_noops_dinotrue_coarse_true_fine_true_vlm_qwen2_5_vl_mt_metavla_rs_libero_lr_5e-5_bs_16_gas_1_ws_0.1_ms_60000_ia_true_acl_4_acm_1_neq_4_ant_layernorm/steps_60000
port=8887
model_family=metavla
device=0

python experiments/robot/server_deploy/deploy.py \
        --pretrained_checkpoint $pretrained_checkpoint \
        --model_family $model_family \
        --port $port \
        --device $device
