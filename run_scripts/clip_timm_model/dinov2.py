# resize_pos_embed_once.py
import timm, torch, os

model_name = 'vit_large_patch14_reg4_dinov2.lvd142m'          # 同理可换 siglip 版本
checkpoint_path = '/inspire/hdd/global_user/chengdongzhou-240108390137/ai_models/timm/vit_large_patch14_reg4_dinov2.lvd142m/pytorch_model.bin'
target_res = 224

# 1. 让 timm 建好 224 结构，同时把原始权重插值进来
model = timm.create_model(model_name, pretrained=True, img_size=target_res, num_classes=0)

# 2. 保存插值后的 state_dict
torch.save(model.state_dict(), checkpoint_path)
print('saved ->', os.path.abspath(checkpoint_path))