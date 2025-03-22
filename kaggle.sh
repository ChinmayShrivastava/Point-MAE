#!/bin/bash

pip install --user kaggle

mkdir -p ~/.kaggle

cp kaggle.json ~/.kaggle/

chmod 600 ~/.kaggle/kaggle.json

kaggle datasets list --mine

kaggle datasets download -d chinmayshri/cleaned-point-cloud

# export LD_LIBRARY_PATH=$(python -c "import torch; print(f'{torch.__path__[0]}/lib')"):$LD_LIBRARY_PATH