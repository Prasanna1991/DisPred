import argparse
import csv

import pickle
import numpy as np
from itertools import compress
import pandas as pd
import math
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import torch

class GenomeDataset(Dataset):
    def __init__(self, data, targets, ETTarget=None):
        self.data = torch.Tensor(data) 
        self.targets = torch.Tensor(targets)
        self.ETTarget = ETTarget
        if not self.ETTarget is None:
            self.ETLabel = torch.Tensor(ETTarget)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if not self.ETTarget is None:
            y1 = self.ETTarget[index]
            return x, y, y1
        else:
            return x, y
    
    def __len__(self):
        return len(self.data)

def getXBP(group, endIn = 52, onlyBP=False, ADSP = False):
      path = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/" + group + "/FromR"
      print(path)
      BP_total = []
      for i in range(1, endIn): #503
            if not onlyBP:
                  X_file = path + "/myMat_{}.npy".format(i)
                  X = np.load(X_file)

            BP_file = path + "/myBP_{}.npy".format(i)
            BP = np.squeeze(np.load(BP_file)).tolist()

            commonElement = set(BP).intersection(BP_total)
            toAddElements = [item for item in BP if item not in commonElement]
            toAddIndx = [BP.index(x) for x in toAddElements]

            #add the okay BP in the list
            BP_total.extend(toAddElements)
            
            if not onlyBP:
                  if i == 1:
                        X_total = X[:, toAddIndx]
                  else:
                        X = X[:, toAddIndx]
                        X_total = np.concatenate((X_total, X), axis=1)
      
                  #Remove nan
                  mask = np.all(~np.isnan(X_total), axis=0)
                  X_total = X_total[:, mask]
                  BP_total = list(compress(BP_total, mask.tolist()))

      #check for repetetion (if any remove them)
      temp1 = list(set([x for x in BP_total if BP_total.count(x) == 1]))
      temp2 = list(set([x for x in BP_total if BP_total.count(x) > 1]))
      temp1.extend(temp2)

      if len(temp1) < len(BP_total):
        newBP = [-1.0] * len(temp1)
        if not onlyBP:
            newX = np.random.rand(X_total.shape[0], len(temp1))
        for i in range(len(temp1)):
            k = BP_total.index(temp1[i])
            newBP[i] = BP_total[k]
            if not onlyBP:
                newX[:, i] = X_total[:, k]
        
        if not onlyBP:
            X_total = newX.copy()
            del newX
        BP_total = newBP.copy()
        del newBP
      
      if not onlyBP:
            #Load target
            if ADSP:
                  y_file = path + "/Y_AD.npy"
                  Y1 = np.load(y_file)
                  y_file = path + "/Y_race.npy"
                  Y2 = np.load(y_file)

                  return X_total, BP_total, Y1, Y2
            else:
                  y_file = path + "/Y.npy"
                  Y = np.load(y_file)

                  return X_total, BP_total, Y

      else: 
            return BP_total

def getDataLoaders(args, data, train_kwargs, test_kwargs, test_size=70000, val_size=70000, normalize=True, UKB_thres = 2.0):
    Y = None
    if data == 1: #full dataset
        x_file = "/oak/stanford/groups/zihuai/Prashnna/data/filteredData/GroupSE1/myMat.npy"
        X = np.lib.format.open_memmap(x_file, dtype='float', mode='r+') 

        x_file = "/oak/stanford/groups/zihuai/Prashnna/data/filteredData/GroupSE2/myMat.npy"
        X = np.concatenate((X, np.lib.format.open_memmap(x_file, dtype='float', mode='r+')), axis=1)

        x_file = "/oak/stanford/groups/zihuai/Prashnna/data/filteredData/GroupSE3/myMat.npy"
        X = np.concatenate((X, np.lib.format.open_memmap(x_file, dtype='float', mode='r+')), axis=1)
    elif data == 2: #APOE imp filtered 
        x_file = "/oak/stanford/groups/zihuai/Prashnna/data/filteredData/APOE/myMat_ImpFiltered.npy" 
        X = np.lib.format.open_memmap(x_file, dtype='float', mode='r+') 
    elif data == 3: #APOE without imp filtering
        x_file = "/oak/stanford/groups/zihuai/Prashnna/data/filteredData/APOE/myMat.npy" 
        X = np.lib.format.open_memmap(x_file, dtype='float', mode='r+') 
    elif data == 4: #full set UKB
        group = "Data1" 
        X, BP, Y = getXBP(group)
    elif data == 5: # ADSP
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_ADSP1/TrainTestData"
        racesN = ["AFR", "EUR", "HIS"]

        trainX = np.load(fileDir + "/trainX_all.npy")
        trainY = np.load(fileDir + "/trainY_all.npy")

        testX = np.load(fileDir + "/testX_all.npy")
        testY = np.load(fileDir + "/testY_all.npy")

        X = trainX
        Y = trainY

    elif data == 6 or data == 7: # ADSP for disentanglement
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_ADSP1/TrainTestData"
        racesN = ["AFR", "EUR", "HIS"]

        ETlabel = []
        trainX = []
        trainY = []
        for i in range(len(racesN)):
            trainX_ = np.load(fileDir + "/trainX_{}.npy".format(racesN[i]))
            trainY_ = np.load(fileDir + "/trainY_{}.npy".format(racesN[i]))

            trainX.append(trainX_)
            trainY.append(trainY_)

            train_ET = np.expand_dims(np.random.rand(trainX_.shape[0]), axis=1)
            train_ET.fill(i)
            ETlabel.append(train_ET)
        
        trainX = np.concatenate( trainX, axis=0)
        trainY = np.concatenate( trainY, axis=0)
        ETlabel = np.concatenate( ETlabel, axis=0)

        # Load test data
        testX = np.load(fileDir + "/testX_all.npy")
        testY = np.load(fileDir + "/testY_all.npy")

        X = trainX
        Y = trainY

    elif data == 17: # ADSP for disentanglement (new data = 3879 dim)
        group = args.group
        seed = args.seedStr
        # Labels are already in 0-1 space.
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/{}/TrainTestData/{}/".format(group, seed) 

        racesN = ["AFR", "EUR", "HIS"]

        ETlabel = []
        trainX = []
        trainY = []
        for i in range(len(racesN)):
            trainX_ = np.load(fileDir + "/trainX_ADSP_{}.npy".format(racesN[i]))
            trainY_ = np.load(fileDir + "/trainY_ADSP_{}.npy".format(racesN[i]))

            trainX.append(trainX_)
            trainY.append(trainY_)

            train_ET = np.expand_dims(np.random.rand(trainX_.shape[0]), axis=1)
            train_ET.fill(i)
            ETlabel.append(train_ET)
        
        trainX = np.concatenate( trainX, axis=0)
        trainY = np.concatenate( trainY, axis=0)
        ETlabel = np.concatenate( ETlabel, axis=0)

        # Load test data
        testX = np.load(fileDir + "/testX_ADSP_all.npy")
        testY = np.load(fileDir + "/testY_ADSP_all.npy")

        X = trainX
        Y = trainY

    elif data == 18: # UKB for disentanglement (new data = 3879 dim)
        group = args.group
        seed = args.seedStr
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/{}/TrainTestData/{}/".format(group, seed) 

        # fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_UKB_ADSP/TrainTestData/UKB_eth1"
        racesN = ["AFR", "EUR", "MIX", "ASN"] 

        ETlabel = []
        trainX = []
        trainY = []
        for i in range(len(racesN)):
            trainX_ = np.load(fileDir + "/trainX_UKB_{}.npy".format(racesN[i]))
            trainY_ = np.load(fileDir + "/trainY_UKB_{}.npy".format(racesN[i]))

            trainX.append(trainX_)
            trainY.append(trainY_)

            train_ET = np.expand_dims(np.random.rand(trainX_.shape[0]), axis=1)
            train_ET.fill(i)
            ETlabel.append(train_ET)
        
        trainX = np.concatenate( trainX, axis=0)
        trainY = np.concatenate( trainY, axis=0)
        ETlabel = np.concatenate( ETlabel, axis=0)

        # Load test data
        testX = np.load(fileDir + "/testX_UKB_all.npy")
        testY = np.load(fileDir + "/testY_UKB_all.npy")

        # dichotomized the labels
        # trainY = (trainY >= 1.0) * 1
        
        # testY = (testY >= 2.0) * 1
        # testY = (testY >= 1.0) * 1

        X = trainX
        Y = trainY

    elif data == 19: # ADSP for disentanglement (new ADSP - ADNI)
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_UKB_ADSP/ADNI/TrainTestData"
        racesN = ["AFR", "EUR", "HIS"]

        ETlabel = []
        trainX = []
        trainY = []
        for i in range(len(racesN)):
            trainX_ = np.load(fileDir + "/trainX_ADSP_{}.npy".format(racesN[i]))
            trainY_ = np.load(fileDir + "/trainY_ADSP_{}.npy".format(racesN[i]))

            trainX.append(trainX_)
            trainY.append(trainY_)

            train_ET = np.expand_dims(np.random.rand(trainX_.shape[0]), axis=1)
            train_ET.fill(i)
            ETlabel.append(train_ET)
        
        trainX = np.concatenate( trainX, axis=0)
        trainY = np.concatenate( trainY, axis=0)
        ETlabel = np.concatenate( ETlabel, axis=0)

        # Load test data
        testX = np.load(fileDir + "/testX_ADSP_all.npy")
        testY = np.load(fileDir + "/testY_ADSP_all.npy")

        X = trainX
        Y = trainY

    print('Data shape: {}'.format(X.shape))


    if Y is None:
        #Load target
        y_file = "/oak/stanford/groups/zihuai/Prashnna/data/filteredData/Label/Y.npy"
        Y = np.load(y_file)
        print('Target shape: {}'.format(Y.shape))

    """
    Split data into training, val and test

    08/15: For now, removing validation for training autoencoding setup.
    """
    if data != 5 and data != 6 and data != 7 and data != 17 and data != 18  and data != 19:
        trainX, testX, trainY, testY = train_test_split(X,Y,test_size=test_size,random_state=0)
    
    idx = np.arange(trainX.shape[0])
    if data == 18:
        strat_Y = (trainY >= 1.0) * 1
    else:
        strat_Y = trainY
    trainX, valX, trainY, valY, trainIdx, testIdx = train_test_split(trainX,trainY,idx,test_size=val_size,random_state=0,stratify=strat_Y)

    if data == 6 or data == 7 or data == 17 or data == 18 or data == 19:
        ETlabel_val = ETlabel[testIdx]
        ETlabel = ETlabel[trainIdx]
        print(ETlabel.shape)
        print(ETlabel_val.shape)

    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    print(valX.shape, valY.shape)


    if normalize:
        """
        Normalize, using train only so that later test data can be normalized in the same way! 
        Note: (couldn't do it using PyTorch)
        """
        mean, std = trainX.mean(axis=0), trainX.std(axis=0)
        trainX = (trainX - mean) / std
        # valX = (valX - mean) / std
        testX = (testX - mean) / std

    """
    Create dataloader
    """
    if data == 7 or data == 17 or data == 18  or data == 19:
        trainDataset = GenomeDataset(trainX, trainY, ETlabel)
    else:
        trainDataset = GenomeDataset(trainX, trainY)
    testDataset = GenomeDataset(testX, testY)
    valDataset = GenomeDataset(valX, valY)

    train_loader = torch.utils.data.DataLoader(trainDataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testDataset, **test_kwargs)
    val_loader = torch.utils.data.DataLoader(valDataset, **test_kwargs)
    feat_dim = X.shape[1] #to define model

    if data == 6 or data == 7 or data == 17 or data == 18  or data == 19:
        return trainX, trainY, valX, valY, train_loader, test_loader, val_loader, feat_dim, ETlabel
    else:
        return train_loader, test_loader, val_loader, feat_dim