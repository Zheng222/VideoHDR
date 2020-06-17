#!/usr/bin/env bash

python scripts/extract_test.py --videos_folder /mnt/video/train_3nd/HDR_4k/ --dataset_folder /mnt/video/Tencent_round3_extracted/train/HDR_4K/

python scripts/extract_test.py --videos_folder /mnt/video/train_3nd/LDR_540p/ --dataset_folder /mnt/video/Tencent_round3_extracted/train/LDR_540p/