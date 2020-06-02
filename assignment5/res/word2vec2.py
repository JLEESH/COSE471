import torch
import random
import argparse
from random import shuffle
from collections import Counter
from huffman import HuffmanCoding


def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)

    context = corpus[max(0, wordID - C):wordID]
    if wordID + 1 < len(corpus):
        context += corpus[wordID + 1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def CBOW_HS(center, context, codes, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # codes : List of Huffman code element (type:list)                      #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out


def CBOW_NS(center, context, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out


def Skipgram_HS(center, context, codes, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # codes : List of Huffman code element (type:list)                      #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out


def Skipgram_NS(center, context, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Index of a contextword (type:int)                           #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out


def word2vec_trainer(ns, corpus, word2ind, freqdict, ind2node,
                     mode="CBOW", dimension=64, learning_rate=0.05, iteration=50000):
    # initialization
    W_emb = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_emb = W_emb.cuda()
    W_out = W_out.cuda()
    window_size = 5

    losses = []
    for i in range(iteration):
        # Training word2vec using SGD
        while True:
            centerWord, contextWords = getRandomContext(corpus, window_size)
            if len(contextWords) == window_size * 2:
                break

        # to be implemented
        centerInd = None
        contextInds = None

        # choose whether use learning rate decay
        # lr = learning_rate * (1 - i / iteration)
        # lr = learning_rate

        if mode == "CBOW":
            if ns == 0:

                # Only use the activated rows of the weight matrix
                nodes = torch.cuda.LongTensor(ind2node[centerInd.item()][0])
                codes = torch.cuda.LongTensor(ind2node[centerInd.item()][1])
                L, G_emb, G_out = CBOW_HS(centerInd, contextInds, codes, W_emb, W_out[nodes])

            else:
                L, G_emb, G_out = CBOW_NS(centerInd, contextInds, W_emb, W_out)

            W_emb[contextInds] -= lr * G_emb
            W_out[nodes] -= lr * G_out
            losses.append(L.item())

        elif mode == "SG":
            if ns == 0:
                nodes = []
                codes = []
                for i in list(centerInd):
                    nodes.append(ind2node[i.item()][0])
                    codes.append(ind2node[i.item()][1])
                # Only use the activated rows of the weight matrix
                for contextInd in contextInds:
                    nodes, codes = ind2node[centerInd]
                    L, G_emb, G_out = Skipgram_HS(centerInd, contextInd, codes, W_emb, W_out[nodes])
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out -= lr * G_out
            else:
                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram_NS(centerInd, contextInd, W_emb, W_out)
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out -= lr * G_out

            losses.append(L.item())
        else:
            print("Unkwnown mode : " + mode)
            exit()

        if i % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print("Loss : %f" % (avg_loss,))
            losses = []

    return W_emb, W_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('subsampling', metavar='subsampling', type=str,
                        help='"Y" for using subsampling, "N" for not')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    ns = args.ns
    mode = args.mode
    subsampling = args.subsampling
    part = args.part
    # Load and tokenize corpus
    print("loading...")
    if part == "part":
        text = open('text8', mode='r').readlines()[0][:10000000]  # Load a part of corpus for debugging
    elif part == "full":
        text = open('text8', mode='r').readlines()[0]  # Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus)
    processed = []
    # Discard rare words
    for word in corpus:
        if frequency[word] > 1:
            processed.append(word)
        else:
            processed.append(" ")

    vocabulary = set(processed)

    # Assign an index number to a word
    word2ind = {}
    i = 0
    for word in vocabulary:
        word2ind[word] = i
        i += 1
    ind2word = {}
    for k, v in word2ind.items():
        ind2word[v] = k

    print("Vocabulary size")
    print(len(word2ind))
    print("Corpus size")
    print(len(processed))

    # Code dict for hierarchical softmax
    freqdict = {}
    for word in vocabulary:
        freqdict[word] = frequency[word]
    codedict = HuffmanCoding().build(freqdict)
    nodedict = {}
    ind2node = {}
    i = 0
    if ns == 0:
        for word in codedict[0].keys():
            code = codedict[0][word]
            s = ""
            nodeset = []
            codeset = []
            for ch in code:
                if s in nodedict.keys():
                    nodeset.append(nodedict[s])
                else:
                    nodedict[s] = i
                    nodeset.append(i)
                    i += 1
                codeset.append(int(ch))
                s += ch
            ind2node[word2ind[word]] = (nodeset, codeset)

    # Training section
    emb, _ = word2vec_trainer(ns, processed, word2ind, freqdict, ind2node,
                              mode=mode, dimension=300, learning_rate=0.025, iteration=320000)
    torch.save([emb, word2ind, ind2word], 'sg.pt')
