# imported some additional libraries
import argparse
from argparse import RawTextHelpFormatter
import random
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# XOR 문제를 해결하기 위해 dataset 만들기.
X_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).float()
y_data = torch.tensor([0, 1, 1, 0]).float()

"""
NPUT	OUTPUT
A	B	A XOR B
0	0	0
0	1	1
1	0	1
1	1	0
"""


class Model(nn.Module):

    def __init__(self, input_size, H1, output_size):
        super().__init__()
        ###############################################################################################################
        #                  TODO : Forward path를 위한 Linear함수 또는 Weight 와 bias를 정의                           #
        ###############################################################################################################
        
        # re-purpose H1 as the number of layers (excluding output)
        self.H1 = H1


        ## define network (biases set to True by default)
        # no hidden layers
        if H1 == 1: 
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size), # weights for input layer
                nn.Sigmoid()
            )

        # one hidden layer
        elif H1 == 2:
            self.layers = nn.Sequential(
                nn.Linear(input_size, input_size), # weights for input layer
                nn.Sigmoid(),
                nn.Linear(input_size, output_size), # weights for hidden layer
                nn.Sigmoid()
            )

        # 3 hidden layers, activation function only at the last layer
        elif H1 == 4:
            self.layers = nn.Sequential()
            for i in range(1, H1):
                self.layers.add_module('lin_layer_' + str(i), nn.Linear(input_size, input_size))
            self.layers.add_module('lin_layer_last', nn.Linear(input_size, output_size)) # weights for last hidden layer
            self.layers.add_module('sigmoid', nn.Sigmoid())

        # output network set-up
        #print(self.layers)

        
        ## view weights
        #with torch.no_grad():
        #    for index, layer in enumerate(self.layers):
        #        try:
        #            print(self.layers[index].weight)
        #        except AttributeError:
        #            pass


        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        # TODO 1. Linear - Sigmoid 의 구조를 가지는 Forward path를 수행하고 결과를 x에 저장                          #

        # TODO 2. Linear - Sigmoid - Linear - Sigmoid 의 구조를 가지는 Forward path를 수행하고 결과를 x에 저장       #

        # TODO 3. Linear - Linear - Linear - Linear - Sigmoid 의 구조를 가지는 Forward path를 수행하고 결과를 x에 저장#
        ###############################################################################################################
        
        # feed forward according to the pre-defined number of layers
        x = self.layers(x)
        
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

        return x

    def predict(self, x):
        return self.forward(x) >= 0.5


####### code modified #########

desc = 'XOR Assignment.'\
        + '\nUsing SGD with hyperparameters lr=0.005 for 20000 iterations reduces the chances of becoming stuck at a wrong solution.'\
        + '\n\ni.e. python(3) XOR.py -s -l 2 -i 20000 -lr 0.005\n'
parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
parser.add_argument('-l', '--layer', default=2, type=int, choices=[1, 2, 4], help='Number of layers excluding output (default: 2).')
parser.add_argument('-s', '--SGD', action='store_const', const=True, default=False, help='Use SGD (default: BGD).')
parser.add_argument('-i', '--iterations', default=2000, type=int, help='Number of epochs(BGD) or iterations(SGD) (default: 2000).')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate (default: 0.01).')
args = parser.parse_args()
H1 = args.layer
SGD = args.SGD
lr = args.learning_rate
iterations = args.iterations
model = Model(2, H1, 1)
#### code modification end ####


##############################################################################################################
#                  TODO : 손실함수(BCELoss)와 optimizer(Adam)를 정의(learning rate=0.01)                     #
##############################################################################################################

# define loss function
loss_func = nn.BCELoss()

# define optimizer with learning rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr) # note: value set using the variable lr

###############################################################################################################
#                                              END OF YOUR CODE                                               #
###############################################################################################################

epochs = iterations # note: value set using the variable iterations
losses = []

# batch train step
for i in range(epochs):
    loss=None
    ##############################################################################################################
    #                         TODO : foward path를 진행하고 손실을 loss에 저장                                   #
    #                               전체 data를 모두 보고 updata를 진행하는 Batch gradient descent(BGD)을 진행   #
    ##############################################################################################################
    
    # zero gradients
    optimizer.zero_grad()
    
    if SGD == False:
        # look at all data (BGD)
        y_hat = model(X_data)

        # calculate loss according to loss function (binary cross entropy)
        loss = loss_func.forward(y_hat, y_data)
    
    elif SGD == True:
        # Test SGD
        pos = random.randint(0, 3)
        y_hat = model(X_data[pos])
        loss = loss_func.forward(y_hat, y_data[pos])
    
    ###############################################################################################################
    #                                              END OF YOUR CODE                                               #
    ###############################################################################################################

    print("epochs:", i, "loss:", loss.item())
    losses.append(loss.item())
    ##############################################################################################################
    #                     TODO : optimizer를 초기화하고 gradient를 계산 후 model을 optimizing                    #
    ##############################################################################################################
    
    # calculate gradients
    loss.backward()

    # optimize model
    optimizer.step()

    ###############################################################################################################
    #                                              END OF YOUR CODE                                               #
    ###############################################################################################################


def cal_score(X, y):
    y_pred = model.predict(X)
    score = float(torch.sum(y_pred.squeeze(-1) == y.byte())) / y.shape[0]

    return score


print('test score :', cal_score(X_data, y_data))
plt.plot(range(epochs), losses)
plt.show()


def plot_decision_boundray(X):
    x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
    y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))

    xx, yy = np.meshgrid(x_span, y_span)

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()

    pred_func = model.forward(grid)

    z = pred_func.view(xx.shape).detach().numpy()

    plt.contourf(xx, yy, z)
    plt.show()


plot_decision_boundray(X_data)