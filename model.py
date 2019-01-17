from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torchvision import  datasets,models, transforms
import torch.nn.functional as F
import time
import os,cv2
from torch.utils.data import Dataset
from PIL import Image
import math
import numpy as np

# 限制使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 定义一个类，需要创建模型的时候，就实例化一个对象
class SplitPlateModel(nn.Module):
    def __init__(self,Load_VIS_URL=None):
        super(SplitPlateModel,self).__init__()
        
        # 使用resnet18作为backbone
        model_ft = models.resnet18(pretrained=False)
        
        # 如果需要 1 channel
        # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 修改最后一层平均池化，padding=1,64/32=2 2<3
        model_ft.avgpool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)             

        # 防止过拟合
        model_ft.dp = nn.Dropout(p=0.3)

        # 选取features部分作为backbone
        self.backbone = nn.Sequential(*list(model_ft.children())[0:9])
        
        # 添加一个fc 层 / 之前采用 nn.Conv2d(in,out,1) 1*1 卷积效果没有 fc 好
        self.out = nn.Linear(in_features=9216, out_features=512, bias=True)
             
        #self.vertical_coordinate1 = nn.Conv2d(512, 32, 1) # 1*1 卷积
        self.vertical_coordinate1 = nn.Linear(in_features=512 , out_features=32, bias=True) # fc x=0 处  y
        #self.vertical_coordinate2 = nn.Conv2d(512, 32, 1) # 1*1 卷积
        self.vertical_coordinate2 = nn.Linear(in_features=512 , out_features=32, bias=True) # fc x=w 处  y
        self.cls = nn.Linear(in_features=512 , out_features=2, bias=True) #nn.Conv2d(512, 2, 1)  # 分类
           
    def forward(self,x):
        x = self.backbone(x)
        flatten = x.view(x.size(0), -1)
        flatten = self.out(flatten)
        vertical_coordinate1 = self.vertical_coordinate1(flatten)
        vertical_coordinate2 = self.vertical_coordinate2(flatten)
        cls = self.cls(flatten)
        
        # 返回 分类、位置信息
        return cls,vertical_coordinate1,vertical_coordinate2
