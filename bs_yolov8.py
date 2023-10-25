# Background subtraction using YoloV8

# Testing
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

import cv2 as cv
import onnxruntime
from ultralytics import YOLO

print('Background subtraction using YoloV8')

#Using pre-trained model, maybe we can train our own? based on a user playing the game
#model = YOLO('yolov8n-seg.pt')
#model.export(format="onnx", imgsz=640)

#Onnx
#session = onnxruntime.InferenceSession('yolov8n-seg.onnx')
model = 'yolov8n-seg.onnx'
yolov8detector = YOLO(model) #seems to run on onnx, w/ gpu

#Capturing video through OpenCV
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Unable to open camera")
    exit()

while True:
    ret, frame = capture.read()

    if not ret:
        print("Frame wasn't read correctly, exiting..")
        break
    
    #YoloV8
    #Run inference on the frame
    #results = model(frame)

    #Visualize
    #annotated_frame = results[0].plot()
    
    #Using onnx for inferencing
    results = yolov8detector(frame)
    annotated_frame = results[0].plot() #zero-index because we use one letter only
    

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_FPS)), (15, 15),
    cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    cv.imshow("Hello", annotated_frame)
    bg = np.zeros(frame.shape)


    print(results[0])

    if(results[0].masks is not None):
        mask_data = results[0].masks[0].data.cpu().numpy().transpose(1, 2, 0)
        mask_data = np.asarray(mask_data)
        mask_data = np.resize(mask_data, (480, 640))
        mask_data *= 255

        masked_frame = frame
        masked_frame = cv.bitwise_and(frame, frame, mask_data)
       
        print(mask_data.shape)
        print(masked_frame.shape) #masked frame has 3 channels

        print('\n')
        print(np.unique(mask_data))
        #np.savetxt("mask_data", mask_data[0])
        print(np.unique(masked_frame))

        cv.imshow('mask', mask_data)
        cv.imshow("img", masked_frame)
        
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
