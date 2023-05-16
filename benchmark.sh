#!/bin/bash
python train.py --head 2 --reduction 1 --expansion 2 --kernel-size 5 --n-blocks 3 --block-depth 3 \
--accelerator cuda \
--precision 16-mixed \
--preprocess-config config/VCTK/preprocess.yaml \
--out-folder base_english2 \
--verbose \
--max_epochs 10
