import argparse
import os
import os.path
import subprocess
from multiprocessing.dummy import Pool as ThreadPool

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--videos_folder", type=str, default='/mnt/video/train_3nd/HDR_4k/', help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, default='/mnt/video/Tencent_round3_extracted/train/HDR_4k/', help='path to the output dataset folder')
args = parser.parse_args()

def extract_frames(videos, inDir, outDir):
    """
    Converts all the videos passed in `videos` list to images.
    Parameters
        videos : list
            name of all video files.
        inDir : string
            path to input directory containing videos in `videos` list.
        outDir : string
            path to directory to output the extracted images.
    Returns
        None
    """

    for video in videos:
        outPath = os.path.join(outDir, os.path.splitext(video)[0])
        if not os.path.isdir(outPath):
            os.makedirs(outPath)
        retn = os.system("ffmpeg -i {} -pix_fmt rgb48 -vsync 0 {}/%4d.png".format(os.path.join(inDir, video), os.path.join(outDir, os.path.splitext(video)[0])))
        if retn:
            print("Error converting file:{}. Exiting.".format(video))

def test():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.makedirs(args.dataset_folder)

    test_l_Path = os.path.join(args.dataset_folder)

    if not os.path.isdir(test_l_Path):
        os.makedirs(test_l_Path)


    # Extract video names
    test_l_videos = sorted(os.listdir(os.path.join(args.videos_folder)))


    # Create train-val-test dataset
    extract_frames(test_l_videos, os.path.join(args.videos_folder), test_l_Path)

test()