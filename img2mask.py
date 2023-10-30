# Create a mask out of an image/folder of images
# Working... in progress

import os
import argparse as ap
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image

parser = ap.ArgumentParser(description="Use yolov8 object segmentation to mask people")
parser.add_argument("-file", "--file", default=None, help="Filename")
parser.add_argument("-folder", "--folder", default=None, help="")

args = vars(parser.parse_args())
print(args)

#Create an output folder
currentdir = os.getcwd()
outputdir = os.path.join(currentdir, "output")
if not os.path.isdir(outputdir):
    os.mkdir('output')

class Segmentor:
    # model = None
    # def __init__(Segmentor):
    def segment(image):
        model = YOLO('yolov8x-seg.onnx')
        segmentResult = model.predict(image, project='Conversion', name='Result')
        print(segmentResult)
        return segmentResult

process = Segmentor

if args["file"]:
    # print(args["file"])
    path = args["file"]

    if os.path.isfile(path):
        file = os.path.basename(path)

        if file.endswith(('.jpg', '.png')):
            #Read Image, use YoloV8 to extract masks
            data = cv.imread(path)
            result = process.segment(data)[0]
            mask = process.segment(data)[0].masks[0].data.cpu().numpy()
            mask = np.moveaxis(mask, 0, -1)
            print(np.shape(mask), (result.orig_shape))

            #Scale using built-in utils in YoloV8
            mask = scale_image(mask, result.orig_shape)

            #Fill with color white
            color = (255, 255, 255)
            colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
            colored_mask = np.moveaxis(colored_mask, 0, -1)
            black = np.zeros(np.shape(data), np.uint8)

            masked = np.ma.MaskedArray(black, mask=colored_mask, fill_value=color)
            mask_result = masked.filled()

            #display
            cv.namedWindow('Result', cv.WINDOW_GUI_EXPANDED)
            cv.imshow('Result', mask_result)
            cv.waitKey(0)

            #Write file in specified folder
            os.chdir(outputdir)
            newfilename = f'masked_{file}'
            cv.imwrite(newfilename, mask_result)

if args["folder"]:
    path = args["folder"]
    #mult = float(input("Input a float to use for scaling:"))

    for file in os.listdir(path):
        if file.endswith(('.jpg', '.png')):
            with open(os.path.join(path, file)) as image:
                
                data = cv.imread(image.name)
                result = process.segment(data)[0]
                mask = process.segment(data)[0].masks[0].data.cpu().numpy()
                mask = np.moveaxis(mask, 0, -1)
                print(np.shape(mask), (result.orig_shape))

                #Scale using built-in utils in YoloV8
                mask = scale_image(mask, result.orig_shape)

                #Fill with color white
                color = (255, 255, 255)
                colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
                colored_mask = np.moveaxis(colored_mask, 0, -1)
                black = np.zeros(np.shape(data), np.uint8)
                masked = np.ma.MaskedArray(black, mask=colored_mask, fill_value=color)
                mask_result = masked.filled()

                #Write file in specified folder
                os.chdir(outputdir)
                newfilename = f'masked_{file}'
                cv.imwrite(newfilename, mask_result)

                #Go back to previous folder to continue operations
                os.chdir('..')

cv.destroyAllWindows()

