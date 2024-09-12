import torch
import IPython
from IPython.display import Image
import tensorboard
import tensorflow
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
print("k")

# %cd yolov5-master

# !pip install -r requirements.txt

# def detect_yolo(string):
#     !python detect.py --weights runs/train/exp5/weights/last.pt --img 640 --conf 0.25 --source ../string --save-txt
#     exp = 2
#     exp+=exp
#     exp= str(exp)
#     exp = "exp"+exp
#     with open(f'../yolov5/runs/detect/{exp}/labels/img.txt', 'r') as file:
#       # Read the first line of the file
#     line = file.readline()
    
#       # Split the line into individual values
#     values = line.split()
#       # Assign the values to variables
#     cl, x_center, y_center, box_width, box_height = values[0], values[1], values[2], values[3], values[4]
#   return values
