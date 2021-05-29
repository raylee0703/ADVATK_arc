import numpy as np
import torch
import torch.nn as nn

def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label to a one-hot values.

    Parameters
    ----------
        labels : N x H x W, where N is batch size.(torch.Tensor)
        num_classes : int
        device: torch.device information
    -------
    Returns
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    
    labels=labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1) 
    return target

def slice_data(data, w, h):
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第一筆先宣告sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,0,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #切成block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #剩下的n-1筆
            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #疊起來
    sub_data = np.delete(sub_data,0,0) #再刪掉第一筆
    sub_data_R = sub_data [:,np.newaxis,:,:] #加入RGB維度

    #G
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第一筆先宣告sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,1,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #切成block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #剩下的n-1筆
            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #疊起來
    sub_data = np.delete(sub_data,0,0) #再刪掉第一筆
    sub_data_G = sub_data [:,np.newaxis,:,:] #加入RGB維度

    #B
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第一筆先宣告sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,2,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #切成block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #剩下的n-1筆
            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #疊起來
    sub_data = np.delete(sub_data,0,0) #再刪掉第一筆
    sub_data_B = sub_data [:,np.newaxis,:,:] #加入RGB維度

    sub_data = np.concatenate((sub_data_R,sub_data_G),axis=1)
    sub_data = np.concatenate((sub_data,sub_data_B),axis=1)

    return sub_data