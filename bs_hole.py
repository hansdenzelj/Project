# Background subtraction using YoloV8
import numpy as np
import cv2 as cv
import onnxruntime
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
import time

print('Background subtraction using YoloV8')

loadMask = cv.imread('masks/m_pose1.jpg', cv.IMREAD_GRAYSCALE) #load the mask
ret, mask = cv.threshold(loadMask, 50, 255, cv.THRESH_BINARY) #make sure the mask is strictly composed of 0's and 1's

empty = np.zeros(mask.shape, dtype=np.uint8) #black, empty array to generate rgb image
coloredMask = np.invert(mask) #invert 0's and 1's -> 0, 255 (background is white, foreground is black)
coloredMask = cv.merge((empty, coloredMask, empty)) #0, 255, 0 

#Preview of the mask/wall
# preview = cv.imshow('test', coloredMask)
# cv.waitKey(0)
# cv.destroyWindow('test')

def getMaskData(results, frame):
    # mask_data = results[0].masks[0].data.cpu().numpy().transpose(1, 2, 0) #(width, height, color?)
    # mask_data = np.asarray(mask_data) #tuple to array
    # mask_data = cv.resize(mask_data, dsize=(640,640), fx=32, fy=32)
    # mask_data = np.resize(mask_data, (480, 640)) #match opencv resolution
    # mask_data *= 255 #convert to b/w
    # return mask_data
    #Read Image, use YoloV8 to extract masks
    data = frame #640, 480, 3
    
    #Get relevant segmentation Results
    result = results[0] 
    mask = results[0].masks[0].data.cpu().numpy()
    mask = np.moveaxis(mask, 0, -1)

    #Scale using built-in utils in YoloV8
    mask = scale_image(mask, result.orig_shape)

    #Fill subject with white, add background
    color = (255, 255, 255)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    black = np.zeros(np.shape(data), np.uint8)

    masked = np.ma.MaskedArray(black, mask=colored_mask, fill_value=color)
    mask_result = masked.filled()
    return mask_result

detector = YOLO('yolov8n-seg.onnx')
capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    start = time.perf_counter()
    if not ret:
        print("Frame wasn't read correctly, exiting..")
        break

    results = detector(frame)
    #annotated_frame = results[0].plot()

    if results[0].masks is not None:
        mask_data = getMaskData(results, frame) #0, 255
        mask_data = np.array(mask_data, dtype=np.uint8) #0, 255

        mask_data = cv.cvtColor(mask_data, cv.COLOR_BGR2GRAY)
        mask_data = cv.threshold(mask_data, 127, 255, cv.THRESH_BINARY)[1]

        maskbyImage = cv.bitwise_and(mask, mask_data) 
        maskedFrame = cv.bitwise_and(frame, frame, mask=maskbyImage)
        maskbyColor = cv.bitwise_or(coloredMask, maskedFrame)

        #Scoring
        #An array of indexes of the player's pixels
        indices = np.where(mask_data == 255) #returns tuple or row and column
        #Get the number of pixels from image segmentation where the player was detected   
        person_pixels = indices[0].size
        #Take the values of elements in the mask at the indices where pixels where detected of the player
        in_mask = mask[indices[0], indices[1]]
        overlap_count = np.count_nonzero(in_mask == 0) #count the number of elements
        #Get accuracy
        percent = 100 - ((overlap_count / person_pixels) * 100)
        #FPS
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        #Show stuff
        cv.putText(maskbyColor,  f"FPS: {int(fps)}", (10, 40), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
        cv.putText(maskbyColor,  f"Person Pixels: {int(person_pixels)}", (10, 70), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
        cv.putText(maskbyColor,  f"Overlapping Pixels: {int(overlap_count)}", (10, 100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
        cv.putText(maskbyColor,  f"Accuracy: {(percent):.2f}", (10, 130), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
        cv.imshow('frame', maskbyColor)

    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

