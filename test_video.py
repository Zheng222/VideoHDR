import os, argparse, glob

import torch
from model import VideoSR
import utils
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VideoSR tencent test')

    parser.add_argument('--checkpoint', type=str, default='/mnt/zheng/4K_HDR_video_round2_mine_7frames/epoch_899.pth',
                        help='where to save checkpoints')
    parser.add_argument('--nframes', type=int, default=7,
                        help='num frames in input sequence')

    ## dataset options
    parser.add_argument('--test_lr_folder', type=str,
                        default='/data/Tencent_round2_extracted/test/SDR_540p_generated/',
                        help='dataset to test')
    parser.add_argument('--save_folder', type=str,
                        default='/data/Tencent_round2_extracted/test/SDR_4K_0101')
    parser.add_argument('--save_imgs', type=bool, default=True)
    parser.add_argument('--flip_test', type=bool, default=True)

    parser.add_argument("--padding", type=str, default='new_info')
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--n_colors", type=int, default=3)
    parser.add_argument("--n_resgroups", type=int, default=16)
    parser.add_argument("--n_resblocks", type=int, default=10)

    opts = parser.parse_args()
    opts.cuda = True

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    device = torch.device("cuda" if opts.cuda else "cpu")

    ## initialize model
    model = VideoSR.VideoSR(opts)

    ## load pretrained model

    model_dict = utils.load_state_dict(opts.checkpoint)
    model.load_state_dict(model_dict, strict=True)

    ## convert to GPU
    model = model.to(device)

    model.eval()
    border_frame = opts.nframes // 2  # border frames when evaluate

    sub_folder_l = sorted(os.listdir(opts.test_lr_folder))

    sub_folder_name_l = []

    # for each sub-folder
    for sub_folder in sub_folder_l:
        sub_folder_name_l.append(sub_folder)

        save_sub_folder = os.path.join(opts.save_folder, sub_folder)

        img_path_l = sorted(glob.glob(os.path.join(opts.test_lr_folder, sub_folder) + '/*.png'))

        max_idx = len(img_path_l)  # 100

        # if os.path.isdir(save_sub_folder):
        #     continue


        if opts.save_imgs:
            utils.mkdirs(save_sub_folder)

        ### read LR images
        imgs = utils.read_seq_imgs(os.path.join(opts.test_lr_folder, sub_folder))
        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            c_idx = int(os.path.splitext(os.path.basename(img_path))[0]) - 1  # from 0 to 99

            select_idx = utils.index_generation(c_idx, max_idx, opts.nframes, padding=opts.padding)
            # get input images
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(imgs_in)
                output_f = output.data.float().cpu().squeeze(0)

                if opts.flip_test:
                    #flip W
                    output = model(torch.flip(imgs_in, (-1,)))
                    output = torch.flip(output, (-1,))
                    output = output.data.float().cpu().squeeze(0)
                    output_f += output

                    # # flip H
                    # output = model(torch.flip(imgs_in, (-2,)))
                    # output = torch.flip(output, (-2,))
                    # output = output.data.float().cpu().squeeze(0)
                    # output_f += output
                    #
                    # # flip both H and W
                    # output = model(torch.flip(imgs_in, (-2, -1)))
                    # output = torch.flip(output, (-2, -1))
                    # output = output.data.float().cpu().squeeze(0)
                    # output_f += output

                    output_f /= 2

            output = utils.tensor2img(output_f)[:, :, [2, 1, 0]]  # RGB --> BGR

            # save imgs
            if opts.save_imgs:
                cv2.imwrite(os.path.join(save_sub_folder, '{:04d}.png'.format(c_idx + 1)), output)
                print('saved {}-th image'.format(img_idx))
