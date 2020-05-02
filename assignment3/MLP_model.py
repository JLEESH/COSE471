import torch
import torch.nn as nn


class MLP_model(nn.Module):
    def __init__(self):
        super().__init__()

        ##############################################################################################################
        #                         TODO : MLP 모델 생성 (구조는 실험해 보면서 결과가 좋은 것으로 사용할 것)                 #
        ##############################################################################################################
        
        # define the various sizes of the layers
        hidden_size = 200 # required hidden layer size
        output_size = 10 # number of classes
        input_sizes = [32 * 32 * 3, hidden_size * 2, hidden_size, hidden_size, hidden_size, hidden_size, output_size]

        # use a sequential container for the network
        self.layers = nn.Sequential()
        
        # add 5 layers to the sequential container
        for i in range(1, 6):
            layer = nn.Linear(input_sizes[i - 1], input_sizes[i])
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            self.layers.add_module('lin_layer_' + str(i), layer)
            self.layers.add_module('relu_layer_' + str(i), nn.ReLU(inplace=True))
        
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        
        # feed forward according to pre-defined layers
        x = x.view(x.size(0), -1)
        x = self.layers(x)

        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
        return x


