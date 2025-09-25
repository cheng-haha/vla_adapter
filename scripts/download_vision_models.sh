#!/bin/bash
# This script downloads the DINOv2 and SigLIP vision models required for DinoSiglipViT.
# Run this script on a machine with internet access and then move the downloaded files
# to your offline machine.

set -e

# Create a directory to store the models, if it doesn't exist
DOWNLOAD_DIR="pretrained_vision_models"
mkdir -p $DOWNLOAD_DIR

# --- DINOv2 Model ---
# Model name: vit_large_patch14_reg4_dinov2.lvd142m
DINO_URL="https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m/resolve/main/model.safetensors"
DINO_FILENAME="vit_large_patch14_reg4_dinov2.lvd142m.safetensors"
DINO_PATH="$DOWNLOAD_DIR/$DINO_FILENAME"

echo "Downloading DINOv2 model..."
if [ -f "$DINO_PATH" ]; then
    echo "DINOv2 model already exists at $DINO_PATH. Skipping download."
else
    wget -O $DINO_PATH $DINO_URL
    echo "DINOv2 model downloaded to $DINO_PATH"
fi


# --- SigLIP Model ---
# Model name: vit_so400m_patch14_siglip_224
SIGLIP_URL="https://huggingface.co/timm/vit_so400m_patch14_siglip_224/resolve/main/model.safetensors"
SIGLIP_FILENAME="vit_so400m_patch14_siglip_224.safetensors"
SIGLIP_PATH="$DOWNLOAD_DIR/$SIGLIP_FILENAME"

echo "Downloading SigLIP model..."
if [ -f "$SIGLIP_PATH" ]; then
    echo "SigLIP model already exists at $SIGLIP_PATH. Skipping download."
else
    wget -O $SIGLIP_PATH $SIGLIP_URL
    echo "SigLIP model downloaded to $SIGLIP_PATH"
fi


echo -e "\nAll models downloaded successfully to the '$DOWNLOAD_DIR' directory."
echo "Please move this directory and its contents to your offline machine."