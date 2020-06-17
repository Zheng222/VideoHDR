# 一、解题思路说明

## 待解决的问题
低质量540p的SDR视频调色并超分辨率至2160p的HDR视频
官方提供的数据命名为LDR
## 整体思路/方案
分为两步骤进行，先调色，后视频超分辨率

## 数据处理
1. 使用ffmpeg将HDR_4K视频降采样为HDR_540p视频，作为调色部分的GT数据
2. 抽取LDR_540p, HDR_540p的关键帧作为调色网络的训练数据对（以SDR_540p的关键帧为准）
3. 抽取LDR_540p, HDR_4K视频中的全部帧

## 模型训练（损失函数）
损失函数为 ![](https://latex.codecogs.com/svg.latex?{L_1}%20+%20\alpha%20%20\cdot%20DSSIM)
## 主要创新点
1. 使用改进的block-wise non-local模块进行帧间对齐，训练比较稳定。
2. 网络结构采用information multi-distillation block （ACMMM 2019）作为基本模块，整体结构采用RCAN (ECCV 2018) 中residual-in-residual的方式构建极深的重建模块。
3. 先调色，后超分辨率可以显著改善超分辨率的性能。

# 二、数据和模型使用
## 预训练模型的使用情况
无
## 相关论文及预训练模型的下载链接
1. https://github.com/Zheng222/IMDN (IMDN, ACM MM'2019)
2. https://github.com/yulunzhang/RCAN (RCAN, ECCV'2018)


# 三、项目运行环境


## 项目所需的工具包/框架

* scikit-image==0.15.0
* pytorch==1.3.0
* torchvision==0.4.1
* opencv-python==4.1.1.26
* imageio==2.5.0
* kornia==0.1.4
* ffmpeg 4.2.1


## 项目运行的资源环境
* Ubuntu 18.04
* 4卡11G-2080Ti


# 四、项目运行办法


## 项目的文件结构

```
-answer

-dataloading
	-__init__.py
	-color_adjust.py  # 调色模型的数据加载程序
	-common.py 
	-Tencent_dataset.py  # 超分辨率模型的数据加载程序
	
-model
	-Color_ajust  #存放调色模型的临时checkpoint文件
	-SR  # 存放超分辨率模型的临时checkpoint文件

-modules
	-architecture.py
	-block.py
	-loss.py
    -NLmodule.py  # Non-local 模块
    -VideoColor.py  # 调色网络

-pretrain_model

-scripts
	-extract_Iframes.py  # 提取关键帧
	-extract_test.py  # 提取全部帧
	-generate_HDR_540p.py  # 将HDR_4K视频降采样为HDR_540p， 作为调色网络的GT
	-generate_video.py  # 图片转换为视频

-Tencent_round3_extracted
	-test  
		-LDR_540p  # 存放抽帧好的测试数据
		-HDR_540p_generated  # 由调色网络得到的对测试集中的LDR_540p调色的结果
	-train # 存放训练所需的图片
		-LDR_540p
		-HDR_4K
		-HDR_540p_generated  # 由调色网络得到的对训练集中LDR_540p调色的结果，所有帧
		-HDR_540p  # 由原始HDR_4K降采样并抽取关键帧得到的结果，作为调色网络的GT

-test
	
-train_3nd
	-4K
	-540p

-train_3nd_adjust_color  # 由原始HDR_4K降采样得到的HDR_540p视频

-main_adjust_color.py  # 调色网络的训练程序
-main_video.py  # 视频超分辨网络的训练程序
-test1.py  # test1.py | test2.py | test3.py | test4.py为超分辨率网络的测试程序，将测试集分为4部分，分别用4张GPU测试
-test2.py
-test3.py
-test4.py
-test_adjust_color.py  # 调色网络的测试程序
-utils.py  # 工具包


-delete_abnormal_frames.sh  # 删除异常数据集的脚本
-extract_Iframes.sh  # 抽取关键帧的脚本
-generate_HDR_540p_mine.sh  # HDR_4K视频降采样的脚本
-extract_test.sh # 抽取全帧的脚本
```
## 项目的运行步骤
1. 运行 `generate_HDR_540p.sh` 生成HDR_540p分辨率的视频

2. 运行 `extract_test.sh` 将测试视频抽帧

3. 运行 `extract_train.sh` 将训练视频抽帧

4. 运行 `extract_Iframes.sh` 抽取关键帧

5. 运行 `delete_abnormal_frames.sh` 删除异常帧

6. 运行 `python main_adjust_color.py 2>&1 | tee -a color.log` 训练调色网络

7. 运行 `python test_adjust_color.py --test_lr_folder Tencent_round3_extracted/train/LDR_540p/ --output_folder Tencent_round3_extracted/train/HDR_540p_generated/ --checkpoint model/Color_ajust/best_model.pth` （生成已调色的训练图像）

8. `python test_adjust_color.py --test_lr_folder Tencent_round3_extracted/test/LDR_540p/ --output_folder Tencent_round3_extracted/test/HDR_540p_generated/ --checkpoint model/Color_ajust/best_model.pth` (生成已调色的测试图像)

9. 运行 `train_sr.sh` 训练超分辨率网络，在保证GPU利用率及训练速度的情况下，可修改训练脚本中的 `--batch_size` 和 `--patch_size`

10. 同时运行 `python test1.py --checkpoint model/SR/best_model.pth`；`python test2.py --checkpoint model/SR/best_model.pth`；`python test3.py --checkpoint model/SR/best_model.pth`；`python test4.py --checkpoint model/SR/best_model.pth` （test1.py --> os.environ["CUDA_VISIBLE_DEVICES"] = '0',  test2.py --> os.environ["CUDA_VISIBLE_DEVICES"] = '1', ...）

11. 生成视频，运行 `generate_video.sh`



## 运行结果的位置

answer文件夹中


## 时间
调色模型4卡大概10小时左右
超分辨率模型4卡大概70小时
测试如果同时使用test1.py test2.py test3.py test4.py 2小时完成
