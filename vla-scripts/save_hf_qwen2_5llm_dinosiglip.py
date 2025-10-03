'''
Description: 
Date: 2025-10-03 22:06:43
LastEditTime: 2025-10-03 22:10:49
FilePath: \vla_adapter\vla-scripts\save_hf_checkpoint.py
'''
"""
Saves a trained VLA checkpoint in the Hugging Face format.

This script loads a checkpoint from a VLA training run, converts its state dictionary
to be compatible with the Hugging Face OpenVLA implementation, and saves the resulting
model in a format that can be loaded directly using `AutoModelForVision2Seq.from_pretrained`.

Usage:
    python vla-scripts/save_hf_checkpoint.py --vlm_path /path/to/your/training/run/
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
from transformers import AutoConfig, AutoModelForVision2Seq

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models import load, load_vla


@dataclass
class ConvertConfig:
    # fmt: off
    vlm_path: Union[str, Path] = ""                     # Path to the VLM experiment directory.
    output_dir_name: str = "hf_checkpoint"              # Name of the output directory for the HF checkpoint.
    # fmt: on



@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vlm_path = Path(cfg.vlm_path)
    hf_token = ""  # Not used for local checkpoints
    assert "prism-qwen25-extra-dinosiglip-224px-0_5b" in str(vlm_path)
    vlm = load(vlm_path, hf_token=hf_token, load_for_training=True)

    # Instantiate HF-compatible OpenVLA model
    # Note: Assumes `config.json` is located in a standard path relative to `prismatic`
    config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)

    # Remap state dictionary keys for HF compatibility
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

    # Load remapped state dict into HF model
    old_state_dict = vlm.state_dict()
    new_state_dict = rename_state_dict_keys(old_state_dict, replace_map)
    missing_keys, unexpected_keys = vla.load_state_dict(new_state_dict, strict=False)

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    # Save HF-formatted checkpoint
    output_dir = vlm_path / cfg.output_dir_name
    print(f"Saving HF-formatted checkpoint to: {output_dir}")
    vla.save_pretrained(output_dir)

    # Copy all configuration files to the output directory
    configs_path = Path("pretrained_models/configs")
    if configs_path.is_dir():
        print(f"Copying configuration files from {configs_path} to {output_dir}...")
        for file_path in configs_path.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, output_dir)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()