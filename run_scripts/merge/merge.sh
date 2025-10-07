#========== Basic Settings ==========#
PROJECT_PATH=VLA-Adapter
ROOT_PATH=/inspire/hdd/global_user/chengdongzhou-240108390137
export WANDB_CONSOLE=off
export WANDB_MODE=offline
export PYTHONPATH=$ROOT_PATH/vla_projects/$PROJECT_PATH


vlm_path=/inspire/hdd/global_user/chengdongzhou-240108390137/ai_models/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b
lora_finetuned_checkpoint_dir=/inspire/hdd/global_user/chengdongzhou-240108390137/vla_projects/VLA-Adapter/outputs/libero_10_no_noops/sim-p2q_img2_miniTrue_propTrue_proTrue_filmFalse-20251006_133806/configs+libero_10_no_noops+b16+lr-0.0002+MS-150005-DS-150005+lora-r64+dropout-0.0--image_aug--sim-p2q--150000_chkpt
python vla-scripts/merge_lora_weights_and_save.py --vlm_path  $vlm_path --lora_finetuned_checkpoint_dir $lora_finetuned_checkpoint_dir