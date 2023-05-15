#!/bin/bash
python train.py --head 2 --reduction 1 --expansion 2 --kernel-size 5 --n-blocks 3 --block-depth 3 \
--accelerator cuda \
--preprocess-config config/VCTK/preprocess.yaml \
--out-folder base_english2 \
--num-workers 12 \
--verbose \
--max_epochs 60
