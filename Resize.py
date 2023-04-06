import random
import math
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
import os
import re

all_file_name = []
img_file_path = "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/pipe_fryum/train/good"
all_file_name = os.listdir(img_file_path)

class_name= img_file_path.split('/',9)
os.mkdir(class_name[6])
source_image_path = "C:/Users/MVCLAB/Desktop/tools/Resize"
save_image_path = os.path.join( source_image_path + '/',class_name[6])

for i in all_file_name :
    image_path = os.path.join(img_file_path +'/', i)
    img = Image.open(image_path)
    resize_image = transforms.Resize([448,448])
    img = resize_image(img)

    image_np = np.array(img)
    ConvertToNP_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite( save_image_path + '/'+ i,ConvertToNP_image)
