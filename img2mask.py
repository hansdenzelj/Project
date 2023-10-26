# Create a mask out of an image/folder of images
# Work in progress

import os
import argparse as ap

import numpy as np
import cv2 as cv

parser = ap.ArgumentParser(description="Use yolov8 object segmentation to mask people")
parser.add_argument("-file", "--file", default=None, help="Filename")
parser.add_argument("-folder", "--folder", default=None, help="")

args = vars(parser.parse_args())
print(args)

if args["file"]:
    print(args["file"])

if args["folder"]:
    path = args["folder"]
    mult = float(input("Input a float to use for scaling:"))

    for file in os.listdir(path):
        if file.endswith(('.jpg', '.png')):
            with open(os.path.join(path, file)) as image:
                print(image)
                #imgdata = Image.open(image.name)
                #print(imgdata)
                data = cv.imread(image.name)
                height = int(data.shape[0] * mult)
                width = int(data.shape[1] * mult)
                print(width, height)
                data = cv.resize(src=data, dsize=(width, height), interpolation= cv.INTER_LINEAR)
                cv.namedWindow('display', cv.WINDOW_KEEPRATIO)
                cv.resizeWindow('display', 640, 480)
                cv.imshow('display', data)

    cv.waitKey(0)
    cv.destroyAllWindows()

