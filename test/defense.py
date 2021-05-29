import numpy as np
import torch 
import torch.nn as nn
import os
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import copy
import pickle

from optparse import OptionParser

from utils.magnet.util import make_one_hot
from utils.magnet.dataset import carstreamDataset
from net.defense_models import autoencoder,autoencoder2
from utils.magnet.loss import dice_score
from utils.magnet.worker import AEDetector, SimpleReformer, Classifier, operate, filters
from numba import jit
from PIL import Image
import time

batch_size = 1
display_steps = 10


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='../dataset/Attacked', help='data path')
    parser.add_option('--model_path', dest='model_path',type='string',
                      default='checkpoints/', help='model_path')
    parser.add_option('--reformer_path', dest='reformer_path',type='string',
                      default='checkpoints/', help='reformer_path')
    parser.add_option('--detector_path', dest='detector_path',type='string',
                      default='checkpoints/autoencoder1.pth', help='detector_path')
    parser.add_option('--classes', dest='classes', default=28, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=416, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=416, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--reformer', dest='reformer', type='string',
                      help='reformer name(autoencoder1 or autoencoder2)')
    parser.add_option('--detector', dest='detector', type='string',
                      help='detector name(autoencoder1 or autoencoder2)')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--model_device1', dest='model_device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--model_device2', dest='model_device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--model_device3', dest='model_device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--model_device4', dest='model_device4', default=-1, type='int',
                      help='device4 index number')
    parser.add_option('--defense_model_device', dest='defense_model_device', default=0, type='int',
                      help='defense_model_device gpu index number')

    (options, args) = parser.parse_args()
    return options

def test(model, args):
    
    data_path = args.data_path
    n_channels = args.channels
    data_width = args.width
    data_height = args.height
    gpu = args.gpu
    
    # Hyper paremter for MagNet
    thresholds = [0.0001]
    num_pass = 10
    

    detector_model = autoencoder(n_channels)




    print('detector model')
    summary(detector_model, input_size=(n_channels, data_height, data_width), device = 'cpu')

    # set device configuration
    device_ids = []
    
    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device_defense = torch.device(args.defense_model_device)
        
        device_ids.append(args.model_device1)
        
        if args.model_device2 != -1 :
            device_ids.append(args.model_device2)
            
        if args.model_device3 != -1 :
            device_ids.append(args.model_device3)
        
        if args.model_device4 != -1 :
            device_ids.append(args.model_device4)
        
    else :
        device_defense = torch.device("cpu")
    
    detector = AEDetector(detector_model, device_defense, args.detector_path, p=2)

    # set testdataset
    train_set = carstreamDataset(data_path)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size)

    # Defense with MagNet
    print('test start')
    
    for thrs in thresholds :
        
        print('----------------------------------------')

        avg_score = 0.0
        thrs = torch.tensor(thrs)
        sums = 0
        total_time = 0
        killed = 0
        markss = []
        image_count = 0

        with torch.no_grad():
            for inputs in train_loader:
                start = time.time()
                print(os.listdir(data_path)[image_count]) #印出檔名

                all_pass, sums = filters(detector, inputs, thrs, sums) #印出結果 寫在這裡面
                markss.append(sums)
                
                
                if len(all_pass) <= num_pass :
                    total_time += (time.time() - start)
                    killed += 1
                    print('***** Kill',os.listdir(data_path)[image_count],'*****')
                    print()
                    #print('time cost:%.6f' % (time.time() - start))
                    image_count +=1 
                    continue
            
                
                total_time += (time.time() - start)
                image_count +=1 
                print()
            
            print('Results:',len(train_loader),'data ,killed' , killed ,'data')
                
        
    
    
if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    model = None
    
    test(model, args)
    
    