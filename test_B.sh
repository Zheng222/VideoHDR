#!/bin/bash

## convert testB video to frames
python scripts/extract_test.py --videos_folder testB/ --dataset_folder Tencent_round2_extracted/testB/LDR_540p/

## adjusting the color of testB frames
python test_adjust_color.py --test_lr_folder Tencent_round2_extracted/testB/LDR_540p/ --output_folder Tencent_round2_extracted/testB/SDR_540p_generated/ --checkpoint model/Color_adjust/best_model.pth

## inference
python3 test1.py --test_lr_folder Tencent_round2_extracted/testB/SDR_540p_generated/ --output_folder Tencent_round2_extracted/testB/SDR_4K_0109 --checkpoint model/SR/best_model.pth &
python3 test2.py --test_lr_folder Tencent_round2_extracted/testB/SDR_540p_generated/ --output_folder Tencent_round2_extracted/testB/SDR_4K_0109 --checkpoint model/SR/best_model.pth &
python3 test3.py --test_lr_folder Tencent_round2_extracted/testB/SDR_540p_generated/ --output_folder Tencent_round2_extracted/testB/SDR_4K_0109 --checkpoint model/SR/best_model.pth &
python3 test4.py --test_lr_folder Tencent_round2_extracted/testB/SDR_540p_generated/ --output_folder Tencent_round2_extracted/testB/SDR_4K_0109 --checkpoint model/SR/best_model.pth
