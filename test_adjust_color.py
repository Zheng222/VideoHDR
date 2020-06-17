import argparse
import torch
import os
import utils
import glob
from modules import architecture
import cv2  # write

# Testing settings
parser = argparse.ArgumentParser(description='Color-adjust')
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser.add_argument("--test_lr_folder", type=str, default='Tencent_round3_extracted/test/LDR_540p/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='Tencent_round3_extracted/test/HDR_540p_generated/')
parser.add_argument("--checkpoint", type=str, default='model/Color_adjust/best_model.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_lr_folder

model = architecture.ColorNet()
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)
model = model.to(device)

sub_folder_list = sorted(os.listdir(opt.test_lr_folder))
sub_folder_name_list = []

# for each sub-folder
for sub_foler in sub_folder_list:
    sub_folder_name_list.append(sub_foler)
    save_sub_folder = os.path.join(opt.output_folder, sub_foler)

    img_path_l = sorted(glob.glob(os.path.join(opt.test_lr_folder, sub_foler) + '/*.png'))

    if not os.path.isdir(save_sub_folder):
        utils.mkdirs(save_sub_folder)

    ### read LR images
    imgs = utils.read_seq_imgs(os.path.join(opt.test_lr_folder, sub_foler))

    # process each image
    for img_idx, img_path in enumerate(img_path_l):
        # get input images
        img_in = imgs[img_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_in)
            output_f = output.data.float().cpu().squeeze(0)

        output = utils.tensor2np(output_f)[:, :, [2, 1, 0]]
        cv2.imwrite(os.path.join(save_sub_folder, '{:04d}.png'.format(img_idx + 1)), output)
        print("Saved {}-th image".format(img_idx + 1))
