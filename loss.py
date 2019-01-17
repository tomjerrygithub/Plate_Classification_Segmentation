from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import torch.nn.functional as F
import time
import os,cv2
from torch.utils.data import Dataset

class SplitPlateLoss(torch.nn.Module):
    
    def __init__(self):
        super(SplitPlateLoss,self).__init__()
        self.Ls_cls = nn.CrossEntropyLoss()
        self.Ls_cls_v1 = nn.CrossEntropyLoss()
        self.Ls_cls_v2 = nn.CrossEntropyLoss()
        self.Lv_reg = nn.MSELoss() #.SmoothL1Loss() #
    
    def forward(self,score, vertical_pred1,vertical_pred2,label,split_y1,split_y2):
        clsloss = self.Ls_cls(score,label)
        cls_y1 = self.Ls_cls(vertical_pred1,split_y1)
        cls_y2 = self.Ls_cls(vertical_pred2,split_y2)
        #print(split_y1,split_y2)
        print("clsloss: ",clsloss.item(),"cls_y1: ",cls_y1.item(),"cls_y2: ",cls_y2.item())
        totloss = 0.1*clsloss  + 2.0*cls_y1 + 2.0*cls_y2  # 不同的权重
        return totloss
