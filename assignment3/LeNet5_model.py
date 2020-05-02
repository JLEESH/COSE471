import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5_model(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################################################
        #                         TODO : LeNet5 모델 생성                                                             #
        ##############################################################################################################
        
        # define layers as shown in the PDF file
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fcn1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.frelu1 = nn.ReLU(inplace=True)
        self.fcn2 = nn.Linear(in_features=120, out_features=84)
        self.frelu2 = nn.ReLU(inplace=True)
        
        # gaussian connections replaced with FCN + CrossEntropyLoss
        self.fcn3 = nn.Linear(in_features=84, out_features=10)

        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        
        # feed forward
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fcn1(x)
        x = self.frelu1(x)
        x = self.fcn2(x)
        x = self.frelu2(x)
        x = self.fcn3(x)

        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
        return x


