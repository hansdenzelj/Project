# Background subtraction using YoloV8
import numpy as np
import cv2 as cv
import onnxruntime
from ultralytics import YOLO

print('Background subtraction using YoloV8')

# mask = cv.imread('masks/pose1_mask.jpg', 2)
# ret, bwimg = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

#bw = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
# cv.imshow("binary", bwimg)
# cv.waitKey(0)
# cv.imwrite('masks/pose1_new.jpg', bwimg)
# cv.destroyAllWindows()

loadMask = cv.imread('masks/pose1_new.jpg', cv.IMREAD_GRAYSCALE) #load the mask
ret, mask = cv.threshold(loadMask, 50, 255, cv.THRESH_BINARY) #make sure the mask is strictly composed of 0's and 1's

empty = np.zeros(mask.shape, dtype=np.uint8) #black, empty array to generate rgb image
coloredMask = np.invert(mask) #invert 0's and 1's -> 0, 255 (background is white, foreground is black)
coloredMask = cv.merge((empty, coloredMask, empty)) #0, 255, 0 

#Preview of the mask/wall
preview = cv.imshow('test', coloredMask)
cv.waitKey(0)
cv.destroyWindow('test')

def getMaskData(results):
    mask_data = results[0].masks[0].data.cpu().numpy().transpose(1, 2, 0) #(width, height, color?)
    mask_data = np.asarray(mask_data) #tuple to array
    mask_data = cv.resize(mask_data, dsize=(640,640), fx=32, fy=32)
    mask_data = np.resize(mask_data, (480, 640)) #match opencv resolution
    mask_data *= 255 #convert to b/w
    return mask_data

detector = YOLO('yolov8n-seg.onnx')
capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()

    if not ret:
        print("Frame wasn't read correctly, exiting..")
        break

    results = detector(frame)
    #annotated_frame = results[0].plot()

    if results[0].masks is not None:
        mask_data = getMaskData(results) #0, 255
        mask_data = np.array(mask_data, dtype=np.uint8) #0, 255

        #masked_frame = cv.bitwise_and(frame, frame, mask_data) #does not work, apply mask to the image
        # masked_frame[mask_data == 0] = [0, 0, 0] #cheat

        maskbyImage = cv.bitwise_and(mask, mask_data) 
        maskedFrame = cv.bitwise_and(frame, frame, mask=maskbyImage)
        maskbyColor = cv.bitwise_or(coloredMask, maskedFrame)

        #Scoring
        #Determine how many pixels a person has, compare with the amount of pixels overlapping
        # person_pixels = np.count_nonzero(mask_data == 255)

        # print(np.unique(mask_data))
        # p_indices = np.where(mask_data == 255) #a person detected, 255
        # mask_values = np.take(mask, p_indices) #take the values of elements in the mask at the indices 
        # print(mask_values)
        # overlap_count = mask_values.size #count the number of elements 

        #Show stuff
        cv.imshow('mask_data', mask_data)
        cv.imshow('masked_frame', maskedFrame)
        cv.imshow('test', maskbyImage)
        cv.imshow('frame', maskbyColor)

    #print("person: ", person_pixels)
    #print("overlap: ", overlap_count)

    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

