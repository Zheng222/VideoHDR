#!/usr/bin/env bash
# 10099858
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/10099858
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/10099858
# 15922480
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/15922480
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/15922480

# 31545121
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/31545121
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/31545121

# 32490669
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/32490669
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/32490669

# 38001368
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/38001368
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/38001368

# 78821035
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/78821035
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/78821035

# 97231911
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/97231911
rm -rf /mnt/video/Tencent_round3_extracted/train/HDR_4k/97231911



python main_video.py --epochs 1500 --step_size 500 --start-epoch 50 --pretrained model/SR/best_model.pth
python main_video.py --batch_size 64 --epochs 2500 --workers 24 --start-epoch 1500 --pretrained model/SR/best_model.pth --patch_size 224 2>&1 | tee -a sr.log
python main_video.py --batch_size 64 --epochs 4500 --workers 24 --start-epoch 1000 --pretrained model/SR/best_model.pth --patch_size 224 2>&1 | tee -a sr.log
