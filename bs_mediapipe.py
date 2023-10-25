# import numpy as np
# import cv2 as cv
# import mediapipe as mp
# VisionRunningMode = mp.tasks.vision.RunningMode
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# from datetime import datetime

# print('Background subtraction using MediaPipe')

# #Create MediaPipe ImageSegmenter instance
# model_path = 'models/selfie_multiclass_256x256.tflite'
# base_options = python.BaseOptions(model_asset_path=model_path)
# options = vision.ImageSegmenterOptions(
#     base_options=base_options, 
#     running_mode=VisionRunningMode.VIDEO,
#     output_category_mask=True)

# #Capturing video through OpenCV
# capture = cv.VideoCapture(0)

# if not capture.isOpened():
#     print("Unable to open camera")
#     exit()

# with vision.ImageSegmenter.create_from_options(options) as segmenter:

#     while True:
#         ret, frame = capture.read()

#         if not ret:
#             print("Frame wasn't read correctly, exiting..")
#             break
        
#         #Convert frame into a mediapipe image
#         mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
        
#         segmented_frame = segmenter.segment_async(mp_frame)
#         category_mask = segmented_frame.category_mask
#         #0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others
#         subtracted_frame = category_mask.numpy_view()

#         output_img = (subtracted_frame[0])

#         image_data = mp_frame.numpy_view()
#         fg = np.zeros(image_data.shape, dtype=np.uint8)
#         fg[:] = (255, 255, 255)
#         bg = np.zeros(image_data.shape, dtype=np.uint8)
#         bg[:] = (255, 0, 0)

#         condition = np.stack((category_mask.numpy_view(),)* 3, axis=-1) > 0.2
#         output_img = np.where(condition, fg, bg)

#         cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
#         cv.putText(frame, str(capture.get(cv.CAP_PROP_FPS)), (15, 15),
#         cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

#         cv.imshow("Video", output_img)

#         if cv.waitKey(1) == ord('q'):
#             break

#     capture.release()
#     cv.destroyAllWindows()



# Working but Low FPS
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print('Background subtraction using MediaPipe')

#Create MediaPipe ImageSegmenter instance
model_path = 'models/selfie_multiclass_256x256.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

#Capturing video through OpenCV
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Unable to open camera")
    exit()

with vision.ImageSegmenter.create_from_options(options) as segmenter:

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Frame wasn't read correctly, exiting..")
            break
        
        #Convert frame into a mediapipe image
        mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
        
        segmented_frame = segmenter.segment(mp_frame)
        category_mask = segmented_frame.category_mask
        #0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others
        subtracted_frame = category_mask.numpy_view()

        output_img = (subtracted_frame[0])

        image_data = mp_frame.numpy_view()
        fg = np.zeros(image_data.shape, dtype=np.uint8)
        fg[:] = (255, 255, 255)
        bg = np.zeros(image_data.shape, dtype=np.uint8)
        bg[:] = (255, 0, 0)

        condition = np.stack((category_mask.numpy_view(),)* 3, axis=-1) > 0.2
        output_img = np.where(condition, fg, bg)
        
        cv.rectangle(output_img, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(output_img, str(capture.get(cv.CAP_PROP_FPS)), (15, 15),
        cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        cv.imshow("Video", output_img)

        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
