
"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base VLA-Adapter model. Saves the final checkpoint in the same directory.

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla


def find_checkpoint(checkpoints_dir: Path, module_name: str) -> Path:
    """Finds the latest checkpoint file for a given module in the specified directory."""
    pt_files = sorted(checkpoints_dir.glob(f"{module_name}--*_checkpoint.pt"))
    if not pt_files:
        # Fallback for older checkpoint naming convention
        pt_files = sorted(checkpoints_dir.glob(f"step-*-{module_name}.pt"))
        if not pt_files:
            # Fallback for legacy naming convention
            pt_files = sorted(checkpoints_dir.glob(f"*{module_name}.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No checkpoints found for module '{module_name}' in {checkpoints_dir}")

    checkpoint_path = pt_files[-1]
    print(f"Found latest '{module_name}' checkpoint: {checkpoint_path.name}")
    return checkpoint_path


@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (either openvla/openvla-7b or whichever model you fine-tuned / resumed training from)
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter
    vlm_path: Union[str, Path] = "" 
    use_minivla: bool = True                        # 


    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if cfg.use_minivla:
        hf_token = ''
        vlm_path = Path(cfg.vlm_path)
        vlm = load(
            vlm_path,
            hf_token=hf_token,
            load_for_training=True,
            )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)

        # Manually load action_queries weights
        try:
            action_queries_checkpoint_path = find_checkpoint(Path(cfg.lora_finetuned_checkpoint_dir), "action_queries")
            action_queries_state_dict = torch.load(action_queries_checkpoint_path, map_location="cpu")
            RAW_STATE_DICT["action_queries.weight"] = action_queries_state_dict["weight"]
            print("Successfully loaded 'action_queries' weights.")
        except FileNotFoundError as e:
            print(f"Warning: {e}. 'action_queries' weights will be randomly initialized.")

        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)

    else:
        # Load Model using HF AutoClasses
        print(f"Loading base model: {cfg.base_checkpoint}")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter")).to(
        "cuda"
    )
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(cfg.lora_finetuned_checkpoint_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{cfg.lora_finetuned_checkpoint_dir}")


if __name__ == "__main__":
    main()
