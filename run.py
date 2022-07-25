import argparse
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torch.nn.functional as F
from torch import nn

import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from models import GenomeAE_disentgl
from data import getDataLoaders
from utils import getConstrastiveLoss


def train(args, model, device, train_loader, optimizer, epoch,
            no_ET=False, no_DS=False, hingeLoss=True, mseLoss=False,
                            alpha_ET = 1.0, alpha_DS = 1.0):
    model.train()
    totalLoss = 0
    recon_loss_ = 0
    err_DS = 0
    err_ET = 0
    err_ortho = 0
    for batch_idx, (data1, target1, ET1) in enumerate(train_loader):
        data1, target1, ET1 = data1.to(device), target1.to(device), ET1.to(device)

        z1_d, z1_e, x1_hat = model(data1)

        recon_loss1 = F.mse_loss(x1_hat, data1)
        
        if args.data == 18:
            # dichotomized for the UKB
            target1 = (target1 >= 1.0) * 1
        

        errET = getConstrastiveLoss(z1_e, ET1, device, temperature=args.temp, normalize=args.normalizeF)
        errDS = getConstrastiveLoss(z1_d, target1, device, temperature=args.temp, normalize=args.normalizeF)

        loss = recon_loss1 + alpha_DS * errDS + alpha_ET * errET

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item() 
        recon_loss_ += recon_loss1.item()
        err_DS += errDS.item()
        err_ET += errET.item()

    totalLoss /= len(train_loader.dataset)
    recon_loss_ /= len(train_loader.dataset)
    err_DS /= len(train_loader.dataset)
    err_ET /= len(train_loader.dataset)
        
    return totalLoss, recon_loss_, err_DS, err_ET

def test(args, model, device, test_loader, mode='Test', visualize=False, path=None, epoch=None):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.threeLatent:
                z1_d, z1_e, z1_n, x_hat = model(data)   
            elif args.vae:
                z1_d, z_d_logvar, z1_e, z_e_logvar, x_hat = model(data)  
            else:
                z1_d, z1_e, x_hat = model(data)
            test_loss += F.mse_loss(x_hat, data).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)    
    return test_loss

def latentClassifier(args, net, device, trainX=None, trainY=None, testX=None, testY=None, getAll = True, getZd = False, getZe = False, model = 1, mode='Train'):
    if args.data == 17:
        group = args.group
        seed = args.seedStr
        # Labels are already in 0-1 space.
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/{}/TrainTestData/{}".format(group, seed)
    elif args.data == 18:
        group = args.group
        seed = args.seedStr
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/{}/TrainTestData/{}".format(group, seed)
        # fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_UKB_ADSP/TrainTestData/UKB_eth1"
    elif args.data == 19:
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_UKB_ADSP/ADNI/TrainTestData"
    else:
        fileDir = "/oak/stanford/groups/zihuai/Prashnna/data/filteredAllData/Data_ADSP1/TrainTestData"

    if trainX is None:
        if args.data == 17 or args.data == 19:
            trainX = np.load(fileDir + "/trainX_ADSP.npy")
            trainY = np.load(fileDir + "/trainY_ADSP.npy")
        elif args.data == 18:
            trainX = np.load(fileDir + "/trainX_UKB.npy")
            trainY = np.load(fileDir + "/trainY_UKB.npy")
        else:
            trainX = np.load(fileDir + "/trainX_all.npy")
            trainY = np.load(fileDir + "/trainY_all.npy")

        if mode == 'Train':
            # getting the val set for finding the best representation.
            idx = np.arange(trainX.shape[0])
            trainX, testX, trainY, testY, trainIdx, testIdx = train_test_split(trainX,trainY,idx,test_size=1000,random_state=0,stratify=trainY)
        else:
            if args.data == 17 or args.data == 19:
                testX = np.load(fileDir + "/testX_ADSP_all.npy")
                testY = np.load(fileDir + "/testY_ADSP_all.npy")
            elif args.data == 18:
                testX = np.load(fileDir + "/testX_UKB_all.npy")
                testY = np.load(fileDir + "/testY_UKB_all.npy")
                # testY = (testY >= 2.0) * 1
            else:
                testX = np.load(fileDir + "/testX_all.npy")
                testY = np.load(fileDir + "/testY_all.npy")

        trainY = trainY.squeeze()
        testY = testY.squeeze()

    with torch.no_grad():
        train_zd, train_ze = net.encode(torch.Tensor(trainX).to(device))
        test_zd, test_ze = net.encode(torch.Tensor(testX).to(device))

        train_z = torch.cat([train_zd, train_ze], 1)
        test_z = torch.cat([test_zd, test_ze],1)
        trainList = [train_z, train_zd, train_ze]
        testList = [test_z, test_zd, test_ze]
    
  
    testAUCs = []
    models = []
    for i in range(len(trainList)):

        if model == 1:
            # Build lasso regression
            myModel = linear_model.LassoCV(cv=5, n_alphas=10)
        elif model == 2:
            # Build lasso regression
            myModel = linear_model.LinearRegression()
        elif model == 3:
            # Build Random forest classifier
            myModel = RandomForestClassifier()
        elif model == 4:
            # Build RidgeClassifier
            myModel = linear_model.RidgeClassifier(tol=1e-2, solver="sag")
        elif model == 5:
            # Build Perceptron classifier
            myModel = linear_model.Perceptron(max_iter=50)
        elif model == 6:
            # Build Perceptron classifier
            myModel = linear_model.LogisticRegression(max_iter=1000, random_state=0, class_weight='balanced')
        trainZ = trainList[i].cpu().numpy()

        myModel.fit(trainZ, trainY)

        testZ = testList[i].cpu().numpy()
        testPred = myModel.predict(testZ)

        dichotY = testY
        if args.data == 18:
            dichotY = (dichotY >= 2.0) * 1

        fpr, tpr, _ = roc_curve(dichotY, testPred)     
        roc = auc(fpr, tpr)
        testAUCs.append(roc)
        models.append(myModel)
        del myModel
       
    return testAUCs, models

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='UKB Full set AE Model')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--data', type=int, default=1, metavar='N',
                        help='which data? 1. p < 1e-5, 2. APOE Imp 3. APOE w/o Imp')
    parser.add_argument('--group', type=str, default= 'ADSP_pe5',
                            help='GWAS threshold folder')
    parser.add_argument('--seedStr', type=str, default= 'seed1',
                            help='seed value')
    parser.add_argument('--age', action='store_true', default=False,
                            help='Include age as the feature.')
    parser.add_argument('--agel', action='store_true', default=False,
                            help='Include age linear as the feature.')
    parser.add_argument('--sex', action='store_true', default=False,
                            help='Include sex as the feature.')

    parser.add_argument('--outDir', type=str, default= '/oak/stanford/groups/zihuai/Prashnna/model/AE',
                        help='path to save the model')
    parser.add_argument('--fileName', type=str, default= 'temp1',
                        help='filename for the model')
    parser.add_argument('--schedule', action='store_true', default=False,
                        help='For scheduler')
    parser.add_argument('--test_size', type=int, default=70000, metavar='S',
                        help='total sample size for the test data!')
    parser.add_argument('--val_size', type=int, default=70000, metavar='S',
                        help='total sample size for the val data!')
    parser.add_argument('--imgPath', type=str, default= '/oak/stanford/groups/zihuai/Prashnna/model/AE/Images',
                        help='path to save the model')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='For scheduler')
    parser.add_argument('--normalizeF', action='store_true', default=False,
                        help='For feature normalization!')
    parser.add_argument('--zd_dim', type=int, default=100, metavar='S',
                        help='dimension of zd!')
    parser.add_argument('--ze_dim', type=int, default=20, metavar='S',
                        help='dimension of ze!')
    parser.add_argument('--zn_dim', type=int, default=20, metavar='S',
                        help='dimension of zn!')
    parser.add_argument('--threeLatent', action='store_true', default=False,
                        help='include z_noise along with z_d and z_e!') 
    parser.add_argument('--acti', type=int, default=1, metavar='N',
                        help='which model? 1. Non-linear (ReLU) vs. 2. Linear')

    parser.add_argument('--intermediatePrediction', action='store_true', default=False,
                        help='For scheduler')

    parser.add_argument('--no-DS', action='store_true', default=False,
                        help='disables DS loss')
    parser.add_argument('--no-ET', action='store_true', default=False,
                        help='disables ET loss')
    parser.add_argument('--hingeLoss', action='store_true', default=False,
                        help='disables DS loss')
    parser.add_argument('--mseLoss', action='store_true', default=False,
                        help='disables ET loss')

    # Variations for the alpha for similarity losses!
    parser.add_argument('--rampAlpha', action='store_true', default=False,
                        help='slowly increase alpha')
    parser.add_argument('--rampEpoch', type=int, default=1, metavar='S',
                        help='slowly increase alpha after this epoch!')
    parser.add_argument('--rampFor', type=int, default=50, metavar='S',
                        help='how many epoch do we want to keep increasing alpha and then stop!')
    parser.add_argument('--alphaDS', type=float, default=1.0, metavar='LR',
                        help='initial alpha value, if no ramp, this will be the same throughout!')
    parser.add_argument('--alphaET', type=float, default=1.0, metavar='LR',
                        help='initial alpha value, if no ramp, this will be the same throughout!')
    parser.add_argument('--conditionalE', action='store_true', default=False,
                        help='conditional similarity loss for ethnic features!')
    parser.add_argument('--conditionalD', action='store_true', default=False,
                        help='conditional similarity loss for ethnic features!')
    parser.add_argument('--UKB_thres', type=float, default=2.0, metavar='LR',
                        help='dichotomized threshold for UKB disentanglement disease loss.')

    # Variations for orthogonality loss!
    parser.add_argument('--ortho', action='store_true', default=False,
                        help='add orthogonal loss!')    
    parser.add_argument('--orthoAlpha', type=float, default=1.0, metavar='LR',
                        help='weight to the ortho loss!')

    # variations for the predictive model
    parser.add_argument('--myModel', type=int, default=1, metavar='N',
                        help='predictive model. 1. lasso 2. linear')

    parser.add_argument('--temp', type=float, default=0.07, metavar='LR',
                        help='temperature!')

    parser.add_argument('--fromConfig', action='store_true', default=False,
                        help='Get settings from config.')  
    parser.add_argument('--configNum', type=int, default=1, metavar='S',
                        help='row in hyper-param file, counting from 0') 
    parser.add_argument('--configFileNum', type=int, default=-1, metavar='S',
                        help='hyper param file num!') 

    # VAE
    parser.add_argument('--vae', action='store_true', default=False,
                        help='doing vae or ae!')  

    # Adversarial loss
    parser.add_argument('--adv', action='store_true', default=False,
                        help='doing adv or not!')  
    parser.add_argument('--advIter', type=int, default=5, metavar='LR',
                        help='iterations for adversarial optimization!')
    parser.add_argument('--alphaGP', type=float, default=2.0, metavar='LR',
                        help='weight for GP!')
    parser.add_argument('--alpha_wassterstein', type=float, default=0.5, metavar='LR',
                        help='weight for the wassterstein loss during regular optimization!')

    args = parser.parse_args()
    
    if args.fromConfig:
        args = readConfig(args, args.configNum - 1, num=args.configFileNum)
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    """
    Load your dataset
    """
    trainX, trainY, valX, valY, train_loader, test_loader, val_loader, feat_dim, _ = getDataLoaders(args, args.data, train_kwargs, test_kwargs, test_size=args.test_size, val_size=args.val_size, normalize=args.normalize, UKB_thres=args.UKB_thres)
    
    net = GenomeAE_disentgl(dim=feat_dim, zd = args.zd_dim, ze = args.ze_dim, acti=args.acti).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.schedule:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    minVALEpoch = 0.5
    best_epoch = 1

    trainLoss_recon = []
    trainLoss_DS = []
    trainLoss_ET = []
    valLoss = []

    testAUCs_allZ = []
    testAUCs_z_d = []
    testAUCs_z_e = []
    testAUCs_z_n = []

    temp_alphaET = args.alphaET
    temp_alphaDS = args.alphaDS
    if args.rampAlpha:
        addAlphaET = (1 - temp_alphaET) / (args.epochs - args.rampEpoch)
        addAlphaDS = (1 - temp_alphaDS) / (args.epochs - args.rampEpoch)

    for epoch in range(1, args.epochs + 1):

        if args.rampAlpha:
            if epoch > args.rampEpoch and epoch < (args.rampEpoch + args.rampFor):
                temp_alphaET = temp_alphaET  + addAlphaET
                temp_alphaDS = temp_alphaDS  + addAlphaDS


        trainTotal, train_loss, errDS, errET, err_ortho = train(args, net, device, train_loader, optimizer, epoch, 
                                hingeLoss=args.hingeLoss, mseLoss=args.mseLoss, no_ET=False, no_DS=False,
                                alpha_ET = temp_alphaET, alpha_DS = temp_alphaDS)

        val_loss = test(args, net, device, val_loader, mode='Val', visualize=False, path=args.imgPath, epoch=epoch)

        trainLoss_recon.append(train_loss) 
        trainLoss_DS.append(errDS) 
        trainLoss_ET.append(errET) 
        valLoss.append(val_loss)


    if args.save_model:
        outFile = args.outDir + args.group + "/" + args.fileName + "_" + args.seedStr + ".pt"
        torch.save(net.state_dict(), outFile)

    print("Finished training and saving models!")
    print("Best val: {} at epoch : {}".format(minVALEpoch, best_epoch))

    # Analyze the models on the test set!
    del net
    outFile = args.outDir + args.group + "/" + args.fileName + "_" + args.seedStr + "_bestVal.pt"
    print(outFile)

    net = GenomeAE_disentgl(dim=feat_dim, zd = args.zd_dim, ze = args.ze_dim, acti=args.acti).to(device)
    net.load_state_dict(torch.load(outFile))
    [testAUC_all, testAUC_d, tesetAUC_e], _ = latentClassifier(args, net, device, getAll=True, model=args.myModel, mode='Test')
    print('Test all: {}\nTest z_d: {}\nTest z_e: {}'.format(testAUC_all, testAUC_d, tesetAUC_e))

    print("Finished!")

if __name__ == '__main__':
    main()