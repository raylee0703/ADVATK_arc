from utils.yolov4.yolo import YOLO
from PIL import Image
import os
import time

yolo = YOLO()

#while True:
#Path = '../dataset/ivslab_train/JPEGImages/All/1_40_00.mp4'
Path = '../dataset/ivslab_train/test'


for file in os.listdir(Path):


    try:
        img = Image.open(Path+"/"+file)
        name = file
        if name.endswith(".jpg"):
            name = name.replace('jpg', 'txt')  
            r_image = yolo.detect_image(img,name,Path)   
            print("Filename ="+name)
        elif name.endswith(".png"):
            name = name.replace('png', 'txt')  
            r_image = yolo.detect_image(img,name,Path)   
            print("Filename ="+name)
    except:
        print("PASS!!")

print("Finish!!")
