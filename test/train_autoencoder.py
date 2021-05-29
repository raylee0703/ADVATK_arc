import os
import sys
import torch
import torchvision
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import pickle
from optparse import OptionParser
from utils.magnet.dataset import carstreamDataset
from utils.magnet.worker import slice_data
from net.defense_models import autoencoder,autoencoder2
from PIL import Image
from tqdm import tqdm


#display_steps = 112




def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='../dataset/ivslab_train/JPEGImages/All/1_40_00.mp4', help='data path')
    parser.add_option('--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('--classes', dest='classes', default=5, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=12, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=12, type='int',
                      help='image height')
    parser.add_option('--batch_size', dest='batch_size', default=4, type='int',
                      help='batch size')
    parser.add_option('--model_number', dest='model_number', default=1, type='int',
                      help='autoencoder number(1 or 2)')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--device1', dest='device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--device2', dest='device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--device3', dest='device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--device4', dest='device4', default=-1, type='int',
                      help='device4 index number')

    (options, _) = parser.parse_args()
    return options



def train_net(model, args):

    data_path = args.data_path
    num_epochs = args.epochs
    gpu = args.gpu
    
    reg_strength = 1e-9
    
    
    # set device configuration
    device_ids = []
    
    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device = torch.device(args.device1)
        
        device_ids.append(args.device1)
        
        if args.device2 != -1 :
            device_ids.append(args.device2)
            
        if args.device3 != -1 :
            device_ids.append(args.device3)
        
        if args.device4 != -1 :
            device_ids.append(args.device4)
        
    
    else :
        device = torch.device("cpu")
    
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids = device_ids)
        
    device = torch.device(0)
    device_ids.append(0)
        
    
    model = model.to(device)
    print()
   
    train_set = carstreamDataset(data_path)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)

    
    model_folder = os.path.abspath('./checkpoints')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    
    if args.model_number == 1:
        model_path = os.path.join(model_folder, 'AAAAAAAAAAAA.pth')
    
    elif args.model_number == 2:
        model_path = os.path.join(model_folder, 'autoencoder2.pth')
       
    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    
    loss_history = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
    

        
        loss_sum = 0
        model.train()
        lambda2 = torch.tensor(reg_strength)
        l2_reg = torch.tensor(0.)
        count = 0
        for data in tqdm(train_loader, desc="Epoch %d progress: " % (epoch+1)):
            

            if count > 10:
                break
            train_data = torch.from_numpy(slice_data(np.array(data), args.width, args.height)).float()/255

            train_data = train_data.to(device)

            optimizer.zero_grad()
            print(train_data.shape)
            output = model(train_data)
            loss = criterion(output, train_data)

            for param in model.parameters():
                l2_reg += torch.norm(param).detach().cpu()

            loss += lambda2 * l2_reg

            loss.backward()
            optimizer.step()

            loss_sum += loss
            count += 1
            # if batch_idx % display_steps == 0:
            #     print('    ', end='')
            #     print('batch {:>3}/{:>3}, loss {:.4f}\r'\
            #         .format(batch_idx+1, len(train_loader), loss))
                
        # evalute
        print('Finished epoch {}, saving model.'.format(epoch+1))
        print("Total loss=" , loss_sum)
        print()


        model_copy = copy.deepcopy(model)
        model_copy = model_copy.cpu()

        
        model_state_dict = model_copy.state_dict()
        torch.save(model_state_dict, model_path)
        
    return loss_history

if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    model = None
    
    if args.model_number == 1:
        model = autoencoder(n_channels)
    
    elif args.model_number == 2:
        model = autoencoder2(n_channels)
        
    else :
        print("wrong model number : must be 1 or 2")
        raise SystemExit
    
    print('Training Autoencoder {}'.format(str(args.model_number)))
    
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
        
    loss_history = train_net(model, args)
    
    
    
    loss_folder = os.path.abspath('./checkpoints')
    loss_path = 'autoencoder' + str(args.model_number) + '_validation_losses'
    save_loss_path = os.path.join(loss_folder, loss_path)
    
    # save validation loss history
    with open(save_loss_path, 'wb') as fp:
        pickle.dump(loss_history, fp)
