# Background subtraction using YoloV8
import numpy as np
import cv2 as cv
import onnxruntime
from ultralytics import YOLO

print('Background subtraction using YoloV8')

#Using pre-trained model, 
# maybe we can train our own based on the user playing the game? might be more effective

#model = YOLO('yolov8n-seg.pt') # select and download an existing model from YOLO
#model.export(format="onnx") # export model into onnx format, to use onnx's runtime

#use onnx to utilize gpu instead of cpu
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
    
    #YoloV8 - uncomment to make inferences using the cpu
    #Run inference on the frame
    #results = model(frame)

    #Visualize
    #annotated_frame = results[0].plot()
    
    #Using onnx for inferencing (comment both lines below to disable inferences w/ onnx)
    results = yolov8detector(frame)
    annotated_frame = results[0].plot() #zero-index because we use one letter only
    
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_FPS)), (15, 15),
    cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    cv.imshow("Hello", annotated_frame)

    print(results[0])

    if(results[0].masks is not None): #a mask has been generated
        mask_data = results[0].masks[0].data.cpu().numpy().transpose(1, 2, 0)
        mask_data = np.asarray(mask_data) #tuple to array
        mask_data = np.resize(mask_data, (480, 640)) #match opencv resolution
        mask_data *= 255

        masked_frame = frame
        masked_frame = cv.bitwise_and(frame, frame, mask_data)
       
        print(mask_data.shape)
        print(masked_frame.shape) #masked frame has 3 channels

        print(np.unique(mask_data)) #
        print(np.unique(masked_frame))

        cv.imshow('mask', mask_data)
        cv.imshow("img", masked_frame)
        
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
