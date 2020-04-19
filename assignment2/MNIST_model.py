import torch
import torch.nn as nn


class MNIST_model(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################################################
        #                         TODO : 4-layer feedforward 모델 생성 (evaluation report의 세팅을 사용할 것)           #
        ##############################################################################################################

        # define the various sizes of the layers
        hidden_size = 200 # required hidden layer size
        output_size = 10 # number of classes
        input_sizes = [28 * 28, hidden_size, hidden_size, hidden_size, hidden_size, output_size]

        # use a sequential container for the network
        self.layers = nn.Sequential()
        
        # add 4 layers to the sequential container
        for i in range(1, 5):
            layer = nn.Linear(input_sizes[i - 1], input_sizes[i])
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            self.layers.add_module('lin_layer_' + str(i), layer)
            self.layers.add_module('relu_layer_' + str(i), nn.ReLU())

        
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        
        # feed forward according to pre-defined layers
        x = self.layers(x)

        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
        return x


class Config():
    def __init__(self):
        self.batch_size = 200
        self.lr_adam = 0.0001
        self.lr_adadelta = 0.1
        self.epoch = 100
        self.weight_decay = 1e-02 #1e-03 # follow spreasheet value instead
        self.setting_no = 1 # the setting number to use (set of possible values: (1, 2, 3, 4))