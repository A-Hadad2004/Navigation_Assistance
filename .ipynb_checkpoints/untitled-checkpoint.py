import torch
import IPython
from IPython.display import Image
import tensorboard
import tensorflow
import argparse
from pathlib import Path

%cd yolov5-master
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp7/weights/last.pt', help='model.pt path')
    parser.add_argument('--img', type=int, default=640, help='input image size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--source', type=str, default='../img_sofa1.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    opt = parser.parse_args()

    print(opt)
    !python detect.py --weights {opt.weights} --img {opt.img} --conf {opt.conf} --source {opt.source} --save-txt --save-conf
if __name__ == '__main__':
    main()