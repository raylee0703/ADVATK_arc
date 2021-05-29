import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import os
import copy

def to_img(x):
    x = x.clamp(0, 1)
    return x

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

def slice_data_for_arc(data, w, h):
    """
    sub_data = data[0, 0:1, 0 : h, 0 : w]
    split_data = np.array(np.split(data[0,0,:,:], data.shape[3]/w, axis = 1))
    for j in range(split_data.shape[0]): #切成block
        sub_split_data = np.array(np.split(split_data[j,:,:], int(split_data.shape[1]/h), axis = 0)) #剩下的n-1筆
        sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #疊起來
    sub_data = np.delete(sub_data,0,0)
    sub_data = np.expand_dims(sub_data, axis=1)
    return sub_data
    """
    sub_data = data[0, 0:1, 0 : h, 0 : w] #第一筆先宣告sub_data
    for i in range(data.shape[0]): #batch 
        split_data = np.array(np.split(data[i,0,:,:], data.shape[3]/w, axis = 1))    

        for j in range(split_data.shape[0]): #切成block
            sub_split_data = np.array(np.split(split_data[j,:,:], data.shape[2]/h, axis = 0)) #剩下的n-1筆
            sub_data = np.concatenate((sub_data,sub_split_data),axis=0) #疊起來
    sub_data = np.delete(sub_data,0,0) #再刪掉第一筆
    sub_data = np.expand_dims(sub_data, axis=1)
    return sub_data
class AEDetector:
    def __init__(self, model, device, path, p=1):
        """
        Error based detector.
        Marks examples for filtering decisions.
        
        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model
        self.model.load_state_dict(torch.load(path))
        self.path = path
        self.p = p
        
        self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def mark(self, X):
       
        if torch.is_tensor(X):
            X_torch = X
        else :
            X_torch = torch.from_numpy(X)

        print("x:", X.shape)

        result = self.model(X_torch.to(self.device)).detach().cpu()
            
        diff = torch.abs(X_torch - 
                         result)
        print("result shape:", result.shape)
        marks = torch.mean(torch.pow(diff, self.p), dim = (1,2,3))
        
        return marks

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]
    
class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X

    def print(self):
        return "IdReformer:" + self.path
    
class SimpleReformer:
    def __init__(self, model, device, path):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.

        path: Path to the autoencoder used.
        """
        self.model = model
        self.model.load_state_dict(torch.load(path))
        #self.model = load_model(path)
        self.path = path
        self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
 
    def heal(self, X):
        #X = self.model.predict(X)
        #return np.clip(X, 0.0, 1.0)
 
        if torch.is_tensor(X):
            X_torch = X
        else :
            X_torch = torch.from_numpy(X)
        
        X = self.model(X_torch.to(self.device)).detach().cpu()
        
        return torch.clamp(X, 0.0, 1.0)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]

class Classifier:
    def __init__(self, model, device, classifier_path, device_ids = [0]):
        """
        Keras classifier wrapper.
        Note that the wrapped classifier should spit logits as output.

        model : pytorch model class
        device : torch.device
        classifier_path: Path to Keras classifier file.
        """
        self.path = classifier_path
        
        self.model = model
        self.model.load_state_dict(torch.load(classifier_path))
        
        self.softmax = nn.Softmax(dim = 1)
 
        self.device = device
    
        if len(device_ids) > 1 :
            self.model = nn.DataParallel(self.model, device_ids = device_ids)
    
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def classify(self, X, option="logit", T=1):
        
        if torch.is_tensor(X):
            X_torch = X
        else :
            X_torch = torch.from_numpy(X)
          
        X_torch = X_torch.to(self.device)
        
        if option == "logit":
            
            return self.model(X_torch).detach().cpu()
        if option == "prob":
            logits = self.model(X_torch) / T
            logits =  self.softmax(logits)
            return logits.detach().cpu()
            
    def print(self):
        return "Classifier:"+self.path.split("/")[-1]
    
    
class AttackData:
    def __init__(self, examples, labels, name=""):
        """
        Input data wrapper. May be normal or adversarial.

        examples: object of input examples.
        labels: Ground truth labels.
        """

        self.data = examples
        self.labels = labels
        self.name = name

    def print(self):
        return "Attack:"+self.name

def operate(reformer, classifier, inputs, filtered = True):
    
    X = inputs
 
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    if filtered :
        X_prime = reformer.heal(X)
        Y_prime = classifier.classify(X_prime)
        
    else :
        Y_prime = classifier.classify(X)

    return Y_prime  

def filters(detector, data, thrs, sums, width=60, height=60):
    """
    untrusted_obj: Untrusted input to test against.
    thrs: Thresholds.

    return:
    all_pass: Index of examples that passed all detectors.
    collector: Number of examples that escaped each detector.
    """
    sums = 0
    collector = []
    all_pass = np.array(range(10000))

    # data split into blocks

    data = np.array(data)

    data = torch.from_numpy(slice_data(data, width, height)).float()/255
    
    marks = detector.mark(data)

    np_marks = marks.numpy()


    np_thrs = thrs.numpy()
    
    #print(np_marks)
    #print(np_thrs)

    sums += np_marks
    atk_sum = 0
    cln_sum = 0

    clean_counter = 0
    attack_counter = 0
    for i in range(len(np_marks)):
        if np_marks[i] <= np_thrs:
            #print('Block',i,'is clean.')
            clean_counter += 1
            cln_sum = np_marks[i]
        else:
            #print('Block',i,'is attacked.')
            attack_counter += 1
            atk_sum += np_marks[i]

    print(clean_counter,'clean blocks')
    print(attack_counter,'attacked blocks')
    print("atk_val = ", atk_sum / attack_counter)
    if clean_counter != 0:
        print("cln_val = ", cln_sum / clean_counter)

    idx_pass = np.argwhere(np_marks < np_thrs)
    
    collector.append(len(idx_pass))
    all_pass = np.intersect1d(all_pass, idx_pass)
    #print(all_pass)
    all_pass = torch.from_numpy(all_pass)

    return all_pass, sums