#!/bin/bash

sh generate_HDR_540p_mine.sh && sh extract_test.sh && sh extract_Iframes.sh && sh delete_abnormal_frames.sh &&

python main_adjust_color.py 2>&1 | tee -a color.log &&

python test_adjust_color.py --test_lr_folder /mnt/video/Tencent_round3_extracted/train/LDR_540p/ --output_folder /mnt/video/Tencent_round3_extracted/train/HDR_540p_generated/ --checkpoint model/Color_adjust/best_model.pth &&

python test_adjust_color.py --test_lr_folder /mnt/video/Tencent_round3_extracted/test/LDR_540p/ --output_folder /mnt/video/Tencent_round3_extracted/test/HDR_540p_generated/ --checkpoint model/Color_adjust/best_model.pth &&

sh train_sr.sh
