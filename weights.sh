#!/bin/bash

if [ -f "data/weights/pretrain.pth" ]; then
    echo "Pre-trained weights already exist in data/weights/pretrain.pth"
    exit 0
fi

wget https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth

mkdir -p data/weights
mv pretrain.pth data/weights/

echo "Weights downloaded and saved to data/weights/ directory."