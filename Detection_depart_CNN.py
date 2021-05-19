import os
import cv2
import numpy as np
from tqdm import tqdm
from pyAudioAnalysis import MidTermFeatures as aFm
from pyAudioAnalysis import audioBasicIO as aIO
import matplotlib.pyplot as plt


import librosa
import librosa.display
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')




REBUILD_DATA = False

class BipVsNonbip():
    BIPS = "Bips"
    NONBIPS = "Non_Bips"
    LABELS = {BIPS : 0, NONBIPS : 1}
    training_data = []
    bipcount = 0
    nonbipcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                # try:
                path = os.path.join(label,f)
                fs, s_ref = aIO.read_audio_file(path)
                duration = len(s_ref) / float(fs)
                win, step = 0.05, 0.05
                win_mid, step_mid = duration, 10
                mt_ref, st_ref, mt_n_ref = aFm.mid_feature_extraction(s_ref, fs, win_mid * fs, step_mid * fs,
                                                      win * fs, step * fs)
                self.training_data.append([np.array(mt_ref.T), np.eye(2)[self.LABELS[label]]])
            
                if label == self.BIPS:
                    self.bipcount += 1
            
                elif label == self.NONBIPS:
                    self.nonbipcount += 1
                # except Exception as e:
                #     pass
                
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("nb 1 =",self.bipcount)
        print("nb 2 =",self.nonbipcount)
        
        
        
if REBUILD_DATA:
    bipvsnonbip = BipVsNonbip()
    bipvsnonbip.make_training_data()
    
training_data = np.load("training_data.npy", allow_pickle=True)

# training_data_bis = [[0]*136,[0,0]]*20
# for i in range(len(training_data)):
#     for j in range(len(training_data[0])):
#         training_data_bis[i][j] = training_data[i][j][0]



import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, 5)
        self.conv2 = nn.Conv1d(24, 48, 5)        
        self.conv3 = nn.Conv1d(48, 48, 5)
        self.fc1 = nn.Linear(2400,64)
        self.fc2 = nn.Linear(64,10)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (4,2))
        x = F.relu(self.conv3(x))
        x.view(-1,2400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



net = Net()

import torch.optim as optim





















        
