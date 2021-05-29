from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import os.path
import pickle
import torch
from PIL import Image

def slice_data(data, w, h):
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第�?筆�?�??sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,0,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #?��?block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #?��??�n-1�?            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #?�起�?    sub_data = np.delete(sub_data,0,0) #?�刪?�第一�?    sub_data_R = sub_data [:,np.newaxis,:,:] #?�入RGB維度

    #G
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第�?筆�?�??sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,1,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #?��?block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #?��??�n-1�?            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #?�起�?    sub_data = np.delete(sub_data,0,0) #?�刪?�第一�?    sub_data_G = sub_data [:,np.newaxis,:,:] #?�入RGB維度

    #B
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第�?筆�?�??sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,2,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #?��?block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #?��??�n-1�?            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #?�起�?    sub_data = np.delete(sub_data,0,0) #?�刪?�第一�?    sub_data_B = sub_data [:,np.newaxis,:,:] #?�入RGB維度

    sub_data = np.concatenate((sub_data_R,sub_data_G),axis=1)
    sub_data = np.concatenate((sub_data,sub_data_B),axis=1)

    return sub_data


class carstreamDataset(Dataset):
    
    def __init__(self, root_dir, width=60, height=60):
    
        self.data_path = root_dir
        self.width = width
        self.height = height
        self.transform = None

        self.images = []

#        for filename in os.listdir(self.data_path):
#            if filename.endswith(".jpg"):
#                self.images.append(os.path.join(self.data_path, filename))

        for filename in os.listdir(self.data_path):
            if filename.endswith(".jpg"):
                self.images.append(os.path.join(self.data_path, filename))
            else:
                for filename_in in os.listdir(self.data_path+'/'+filename):
                    if filename_in.endswith(".jpg"):
                        self.images.append(os.path.join(self.data_path+'/'+filename, filename_in))
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_path = self.images[index]
        
        img = Image.open(image_path)
        arr = np.array(img)
        print(arr)

        arr = arr.swapaxes(1,2)
        arr = arr.swapaxes(0,1)
            
        return arr