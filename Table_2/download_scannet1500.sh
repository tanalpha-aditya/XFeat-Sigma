#!/bin/bash

# Define download directory
download_dir="ScanNet1500"
mkdir -p "$download_dir"

# Google Drive file (test images)
echo "Downloading test images..."
gdown --id 1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd -O "$download_dir/test_images.tar"

echo "Extracting test images..."
tar -xf "$download_dir/test_images.tar" -C "$download_dir" && rm "$download_dir/test_images.tar"

# GitHub file (ground truth poses)
echo "Downloading ground truth poses..."
wget -c "https://github.com/zju3dv/LoFTR/raw/refs/heads/master/assets/scannet_test_1500/test.npz" -O "$download_dir/test.npz"

echo "Download and extraction complete."
