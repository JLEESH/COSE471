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
    h = inputMatrix[context].sum(0) / len(context) #size(D)
    y = outputMatrix.matmul(h) #size(len(codes), D) * size(D) = size(len(codes))
    sigmoid_y = torch.sigmoid(y) #size(len(codes))

    
    loss = 1
    for i in range(len(codes)):
        if codes[i] == 0:
            loss *= sigmoid_y[i]
        else:
            loss *= 1 - sigmoid_y[i]
    loss = -loss.log()
    
    t = -torch.tensor(codes).cuda() + 1 #size(len(codes)), [0,1,0,0,1] -> [1,0,1,1,0]
    sigmoid_y_minus_t = sigmoid_y - t #size(len(codes))
    EH = (sigmoid_y_minus_t.unsqueeze(1) * outputMatrix).sum(0) #size(len(codes), 1) x size(len(codes), D) = size(len(codes), D) / after sum: size(D)
    grad_emb = EH.unsqueeze(0) / len(context) #size(1, D)
    grad_out = sigmoid_y_minus_t.unsqueeze(1) * h #size(len(codes), 1) x size(D) = size(len(codes), D)
    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    return loss, grad_emb, grad_out


def CBOW_NS(center, context, inputMatrix, outputMatrix, samples):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################
    h = inputMatrix[context].sum(0) / len(context) #size(D)

    target_vector = outputMatrix[center].dot(h)
    sigmoid_target_vector = torch.sigmoid(target_vector)
    target_loss = -sigmoid_target_vector.log()

    sample_vector = outputMatrix[samples].matmul(h)
    sigmoid_sample_vector = torch.sigmoid(sample_vector)
    sample_loss = -(1 - sigmoid_sample_vector).log().sum(0)

    loss = target_loss + sample_loss
    EH_target = (sigmoid_target_vector - 1) * outputMatrix[center]
    EH_sample = (sigmoid_sample_vector.unsqueeze(1) * outputMatrix[samples]).sum(0)
    EH = EH_target + EH_sample
    grad_emb = EH / len(context)

    grad_out = torch.zeros_like(outputMatrix)
    grad_out[center] = (sigmoid_target_vector - 1) * h
    grad_out[samples] = sigmoid_sample_vector.unsqueeze(1) * h
    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    return loss, grad_emb, grad_out


def Skipgram_HS(center, context, codes, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # codes : List of Huffman code element (type:list)                      #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################
    h = inputMatrix[center] #size(D)
    y = outputMatrix.matmul(h) #size(len(codes), D) * size(D) = size(len(codes))
    sigmoid_y = torch.sigmoid(y) #size(len(codes))

    loss = 1
    for i in range(len(codes)):
        if codes[i] == 0:
            loss *= sigmoid_y[i]
        else:
            loss *= 1 - sigmoid_y[i]
    loss = -loss.log()

    t = -torch.tensor(codes).cuda() + 1 #size(len(codes))
    sigmoid_y_minus_t = sigmoid_y - t #size(len(codes))
    EH = (sigmoid_y_minus_t.unsqueeze(1) * outputMatrix).sum(0) #size(len(codes), 1) x size(len(codes), D) = size(len(codes), D) / after sum: size(D)
    grad_emb = EH.unsqueeze(0) #size(1, D)
    grad_out = sigmoid_y_minus_t.unsqueeze(1) * h #size(len(codes), 1) x size(D) = size(len(codes), D)
    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    return loss, grad_emb, grad_out


def Skipgram_NS(center, context, inputMatrix, outputMatrix, samples):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Index of a contextword (type:int)                           #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################
    h = inputMatrix[center] #size(D)

    target_vector = outputMatrix[context].dot(h)
    sigmoid_target_vector = torch.sigmoid(target_vector)
    target_loss = -sigmoid_target_vector.log()

    sample_vector = outputMatrix[samples].matmul(h)
    sigmoid_sample_vector = torch.sigmoid(sample_vector)
    sample_loss = -(1 - sigmoid_sample_vector).log().sum(0)

    loss = target_loss + sample_loss
    EH_target = (sigmoid_target_vector - 1) * outputMatrix[context]
    EH_sample = (sigmoid_sample_vector.unsqueeze(1) * outputMatrix[samples]).sum(0)
    EH = EH_target + EH_sample
    grad_emb = EH

    grad_out = torch.zeros_like(outputMatrix)
    grad_out[context] = (sigmoid_target_vector - 1) * h
    grad_out[samples] = sigmoid_sample_vector.unsqueeze(1) * h
    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    return loss, grad_emb, grad_out


def word2vec_trainer(ns, corpus, word2ind, freqdict, ind2node, sumsampling,
                     mode="CBOW", dimension=64, learning_rate=0.05, iteration=50000):
    # initialization
    W_emb = torch.randn(len(word2ind), dimension) / (dimension ** 0.5) 
    W_out = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_emb = W_emb.cuda()
    W_out = W_out.cuda()
    window_size = 5

    # negative sampling
    freq = torch.tensor([float(n) for n in list(freqdict.values())]) ** 0.75
    sum_of_frequency = sum(freq)
    p = freq / sum_of_frequency
    cumsum_p = torch.cumsum(p, dim=0).tolist()
    words = list(freqdict.keys())

    def create_negative_samples(sample_number=5):
        samples = random.choices(words, cum_weights=cumsum_p, k=sample_number)
        return [word2ind[sample] for sample in samples]

    # probability for subsampling
    if subsampling == 'Y':
        vocabulary = set(corpus)
        probability = {}
        threshold = 1e-05
        sum_of_freq = sum(freqdict.values())
        for word in vocabulary:
            if(freqdict[word] > 0):
                probability[word] = 1 - (threshold / (freqdict[word] / sum_of_freq)) ** 0.5
    
    losses = []
    for i in range(iteration):
        # Training word2vec using SGD

        # Subsampling
        processed = []
        if subsampling == 'Y':    
            for word in corpus:
                if word != " " and probability[word] > random.random():
                    processed.append(word)
                else:
                    processed.append(" ")
        else:
            processed = corpus

        while True:
            centerWord, contextWords = getRandomContext(processed, window_size)
            if len(contextWords) == window_size * 2:
                break

        # to be implemented
        centerInd = torch.tensor(word2ind[centerWord]) #centerWord or context에 " "가 존재할 가능성?
        contextInds = torch.tensor([word2ind[context] for context in contextWords])

        # choose whether use learning rate decay
        #lr = learning_rate * (1 - i / iteration)
        lr = learning_rate

        if mode == "CBOW":
            if ns == 0:

                # Only use the activated rows of the weight matrix
                nodes = torch.cuda.LongTensor(ind2node[centerInd.item()][0])
                codes = torch.cuda.LongTensor(ind2node[centerInd.item()][1])
                L, G_emb, G_out = CBOW_HS(centerInd, contextInds, codes, W_emb, W_out[nodes])
                W_emb[contextInds] -= lr * G_emb
                W_out[nodes] -= lr * G_out

            else:
                while True:
                    negative_samples = create_negative_samples()
                    if centerInd.item() not in negative_samples:
                        break
                L, G_emb, G_out = CBOW_NS(centerInd, contextInds, W_emb, W_out, negative_samples)
                W_emb[contextInds] -= lr * G_emb
                W_out -= lr * G_out
            losses.append(L.item())

            

        elif mode == "SG":
            if ns == 0:
                nodes = []
                codes = []
                '''
                for i in list(centerInd):
                    nodes.append(ind2node[i.item()][0])
                    codes.append(ind2node[i.item()][1])
                '''
                # Only use the activated rows of the weight matrix
                for contextInd in contextInds:
                    nodes, codes = ind2node[contextInd.item()] #Is this contextInd?
                    L, G_emb, G_out = Skipgram_HS(centerInd, contextInd, codes, W_emb, W_out[nodes])
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out[nodes] -= lr * G_out
            else:
                for contextInd in contextInds:
                    while True:
                        negative_samples = create_negative_samples()
                        if contextInd.item() not in negative_samples:
                            break
                    L, G_emb, G_out = Skipgram_NS(centerInd, contextInd, W_emb, W_out, negative_samples)
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out -= lr * G_out

            losses.append(L.item())
        else:
            print("Unkwnown mode : " + mode)
            exit()

        if i % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print("iter : ", i)
            print("Loss : %f" % (avg_loss,))
            losses = []

    return W_emb, W_out


if __name__ == "__main__":
    '''
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
    '''
    mode = "CBOW"
    ns = 0
    subsampling = "Y"
    part = "full"
    
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
    emb, _ = word2vec_trainer(ns, processed, word2ind, freqdict, ind2node, subsampling,
                              mode=mode, dimension=300, learning_rate=0.025, iteration=320000)
    torch.save([emb, word2ind, ind2word], 'CBOW_HS_sub.pt')
