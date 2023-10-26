# Project
 Background subtraction school project

## Important note
Unsure if **yolov8** supports inferences using the gpu with onnx, this is a workaround.

Replace **autobacked.py** located in the **ultralytics/nn** folder of your ultralytics package installation (with the one provided in this repo).

## Files

bs_mediapipe.py - attempt to conduct background subtraction w/ mediapipe
bs_yolov8.py - attempt to do background subtraction w/ yolov8
bs_hole.py - hole in the wall game-like features with yolov8
masks/... - mask images
.onnx, .pt, .tflite - models

