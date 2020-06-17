import argparse
import os
import os.path

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--videos_folder", type=str, default='/mnt/video/train_3nd/', help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, default='../train_3nd_adjust_color/', help='path to the output dataset folder')
args = parser.parse_args()

def resize_video(videos, inDir, outDir):
    for video in videos:

        retn = os.system("ffmpeg -i {} -s 960x540 -c:v libx265 -pix_fmt yuv420p10le -crf 0 -x265-params colormatrix=bt2020nc:transfer=smpte2084:colorprim=bt2020 -vsync 0 {} -y".format(os.path.join(inDir, video), os.path.join(outDir, video)))
        if retn:
            print("Error converting file:{}. Exiting.".format(video))


def main_resize():
    # Create dataset_folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.makedirs(args.dataset_folder)

    l_Path = os.path.join(args.dataset_folder, "HDR_540p")

    if not os.path.isdir(l_Path):
        os.makedirs(l_Path)

    gt_videos = sorted(os.listdir(os.path.join(args.videos_folder, "HDR_4k")))


    resize_video(gt_videos, os.path.join(args.videos_folder, "HDR_4k"), l_Path)


main_resize()