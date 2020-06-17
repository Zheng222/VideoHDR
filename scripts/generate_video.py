import argparse
import os
import os.path


parser = argparse.ArgumentParser()
parser.add_argument('--imgs_folder', type=str, default='../Tencent_round3_extracted/test/HDR_540p_generated/', help='path to the folder containing Res images')
parser.add_argument('--videos_folder', type=str, default='../answer_1/', help='path to the foldering containing converted vidoes')
args = parser.parse_args()

def convert_videos(imgs, inDir, outDir):
    """
    Converts all the generated images in `imgs` list to videos.

    Parameters
        videos : list
            name of all sequential imgs files
        inDir : string
            path to input dircetory containing imgs in `imgs` list.
        outDir : string
            path to directory to output the converted videos

    Returns
        None -framerate 24000/1001 -vsync 0 -y


    """
    for img in imgs:
        retn = os.system('ffmpeg -framerate 24000/1001 -start_number 1 -i {}/%4d.png -pix_fmt yuv420p10 -an -vcodec libx265 -x265-params colormatrix=bt2020nc:transfer=smpte2084:colorprim=bt2020 -preset slow -b:v 100M {} -y'.format(os.path.join(inDir, img), os.path.join(outDir, img+'.mp4')))
        if retn:
            print("Error converting file:{}. Exiting.".format(img))



def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.videos_folder):
        os.makedirs(args.videos_folder)

    test_Res_path = os.path.join(args.videos_folder)  # h_res video folders
    if not os.path.isdir(test_Res_path):
        os.makedirs(test_Res_path)

    # imgs names
    test_Res_imgs = sorted(os.listdir(args.imgs_folder))

    # Convert to videos
    convert_videos(test_Res_imgs, args.imgs_folder, test_Res_path)

main()