import argparse
import torch
import os
import utils
import glob
import cv2
from modules import architecture
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Testing settings
parser = argparse.ArgumentParser(description='SR_part2')

parser.add_argument("--test_lr_folder", type=str,
                    default='Tencent_round2_extracted/test/SDR_540p_generated/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str,
                    default='Tencent_round2_extracted/test/SDR_4K_0104')
parser.add_argument("--checkpoint", type=str, default='model/SR/epoch_49.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=4,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
parser.add_argument('--flip_test', type=bool, default=True)

parser.add_argument("--n_colors", type=int, default=3)
parser.add_argument("--nf", type=int, default=64)
parser.add_argument("--n_resgroups", type=int, default=16)
parser.add_argument("--n_resblocks", type=int, default=10)
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_lr_folder

model = architecture.RIRN(opt)
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)
model = model.to(device)

sub_folder_list = sorted(os.listdir(opt.test_lr_folder))
sub_folder_name_list = []
length = len(sub_folder_list)

# for each sub-folder
for sub_foler in sub_folder_list[length//4: length//4 * 2]:
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

            if opt.flip_test:
                # flip W
                output = model(torch.flip(img_in, (-1,)))
                output = torch.flip(output, (-1,))
                output = output.data.float().cpu().squeeze(0)
                output_f += output

                output_f /= 2

        output = utils.tensor2np(output_f)[:, :, [2, 1, 0]]  # rgb --> bgr
        cv2.imwrite(os.path.join(save_sub_folder, '{:04d}.png'.format(img_idx + 1)), output)
        print("Saved {}-th image".format(img_idx + 1))
