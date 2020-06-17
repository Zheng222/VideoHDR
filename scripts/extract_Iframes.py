import os
import subprocess
import cv2

def iframes(filename, save_path, i_frames):
    cap = cv2.VideoCapture(filename)
    for frame_no in i_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        outname = os.path.join(save_path, str(frame_no + 1).zfill(4) + '.png')
        cv2.imwrite(outname, frame)
        print(outname)
    cap.release()

def iframes_ffmpeg(filename, save_path, i_frames):
    for frame_no in i_frames:
        outname = os.path.join(save_path, str(frame_no + 1).zfill(4) + '.png')
        os.system('ffmpeg -i {} -pix_fmt rgb48 -vf "select=eq(n\,{})" -vframes 1 {} -vsync 0 -y'.format(filename, frame_no, outname))

if __name__ == '__main__':

    lr_path = "train_3nd_adjust_color/540p_mine"  # SDR_540p_mine
    ldr_path = "/mnt/video/train_3nd/LDR_540p"  # LDR_540p
    lr_save_path = "train_3nd_adjust_color_extracted/train/HDR_540p_mine"  # the gt(label) for color adjust
    ldr_save_path = "train_3nd_adjust_color_extracted/train/LDR_540p"

    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()

    for root, dirs, files in os.walk(lr_path):
        for file in files:
            if os.path.splitext(file)[-1] == '.mp4':
                lr_file = os.path.join(root, file)
                lr_save = os.path.splitext(lr_file.replace(lr_path, lr_save_path))[0]

                ldr_file = lr_file.replace(lr_path, ldr_path)
                ldr_save = os.path.splitext(ldr_file.replace(ldr_path, ldr_save_path))[0]

                for save_path in [lr_save, ldr_save]:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                out = subprocess.check_output(command + [ldr_file]).decode()
                f_types = out.replace('pict_type=', '').split()
                frame_types = zip(range(len(f_types)), f_types)
                i_frames = [x[0] for x in frame_types if x[1] == 'I']
                print(ldr_file)
                print(i_frames)
                if i_frames:
                    iframes_ffmpeg(lr_file, lr_save, i_frames)
                    iframes_ffmpeg(ldr_file, ldr_save, i_frames)

