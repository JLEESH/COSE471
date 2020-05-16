import torch
from random import shuffle
from collections import Counter
import argparse


def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def Skipgram(center, context, inputMatrix, outputMatrix):
################################  Input  ################################
# center : Index of a centerword (type:int)                             #
# context : Index of a contextword (type:int)                           #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out

def CBOW(center, context, inputMatrix, outputMatrix):
################################  Input  ################################
# center : Index of a centerword (type:int)                             #
# context : Indices of contextwords (type:list(int))                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=64, learning_rate=0.05, iteration=50000):

	#initialization
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    window_size = 5

    
    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerWord, contextWords = getRandomContext(corpus, window_size)
        centerInd = None
        contextInds = None
        
        #learning rate decay
        lr = learning_rate*(1-i/iteration)

        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb[contextInds] -= lr*G_emb
            W_out -= lr*G_out
            losses.append(L.item())

        elif mode=="SG":
        	for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                W_emb[centerInd] -= lr*G_emb.squeeze()
                W_out -= lr*G_out
                losses.append(L.item())
        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
        	avg_loss=sum(losses)/len(losses)
        	print("Loss : %f" %(avg_loss,))
        	losses=[]

    return W_emb, W_out

def main():
	# Write your code of data processing, training, and evaluation
	# Full training takes very long time. We recommend using a subset of text8 when you debug
	pass

main()