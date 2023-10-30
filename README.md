# Project
 Background subtraction school project

## Important note
Unsure if **yolov8** supports inferences using the gpu with onnx, this is a workaround for my amd gpu.

Replace **autobacked.py** located in the **ultralytics/nn** folder of your ultralytics package installation (with the one provided in this repo).

## Masking tool
Use img2mask.py to create binary masks from color images (jpg/png), uses YoloV8's **yolov8x-seg** model converted to onnx for segmentation. Creates an output folder which stores generated masks.

Usage: 
- py img2mask.py --file (path/file.jpg)
- py img2mask.py --folder (path/folder)
## Files

bs_mediapipe.py - attempt to conduct background subtraction w/ mediapipe
bs_yolov8.py - attempt to do background subtraction w/ yolov8
bs_hole.py - hole in the wall game-like features with yolov8
masks/... - mask images
.onnx, .pt, .tflite - models

