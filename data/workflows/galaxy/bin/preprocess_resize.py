#!/usr/bin/env python3
import random
import glob
import sys
import argparse
import time
from PIL import Image



def parse_args(args):
    parser = argparse.ArgumentParser(description="Enter description here")
    parser.add_argument(
        "-i","--input_dir",default="",
        help="directory with data"
        )
    parser.add_argument(
        "-o","--output_dir",default="",
        help="directory for outputs"
        )
    return parser.parse_args(args)



def main():
    start = time.time()
    args = parse_args(sys.argv[1:])
    input_dir  = args.input_dir
    all_images = glob.glob(input_dir + "*_class_*.jpg")


    for img_path in all_images:
        img = Image.open(img_path)
        width, height = img.size
        new_width = 256
        new_height = 256

        left   = (width - new_width)/2
        top    = (height - new_height)/2
        right  = (width + new_width)/2
        bottom = (height + new_height)/2

        img = img.crop((left, top, right, bottom))
        img_path = img_path.split(".")[0]
        img_path = img_path + "_proc.jpg"
        img.save(img_path)

    exec_time = time.time() - start
    print('Execution time in seconds: ' + str(exec_time))




if __name__ == '__main__':
	main()


