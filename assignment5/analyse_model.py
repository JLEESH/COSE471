import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import random
import argparse
import pickle
from w2v import Word2Vec


# comment lines out as necessary
#model_filename = ".\models\skipgram_ns_subsampling\w2v_model_skipgram_ns_subsample_lr1_lr0.5_mixed (250K) (ps)"
#model_filename = "w2v_model_skipgram_ns_subsample_lr1"
model_filename = "w2v_analyze_model"

# hacky code to instantiate the Word2Vec class to minimise code editing
self = Word2Vec(architecture="skipgram", mode="negative_sampling", subsample=True,
                model_filename=model_filename, pickle_filename="w2v_vars_process_corpus",
                load_model=True, debug=True, learning_rate=1)
self.subsampling_threshold = 0.0001 # use a different threshold




# modified skipgram_ns for debugging/testing
def skipgram_ns_modified(center_index, context_indices):
    # obtain embedding indices
    center_emb_index = self.word2ind[self.corpus[center_index]]
    context_emb_indices = [self.word2ind[self.corpus[index]]
                            for index in context_indices]
    
    # extract the center vector
    center_vector = self.W_emb[center_emb_index]

    loss = 0
    # perform update for each context word
    for index in context_emb_indices:
    
        # negative sampling
        # obtains a view of W_out with the relevant indices
        indices_list = self.get_ns_indices()
        indices_list.append(index)
        self.W_out_ns = self.W_out[indices_list]

        # feed forward
        output = torch.mv(self.W_out_ns, center_vector)
        softmax = F.softmax(output, 0)

        # calculate loss and gradients
        g = softmax.clone()
        g[-1] -= 1
        loss += (-1 * F.log_softmax(output, 0)[-1]).item()
        grad_emb = torch.mv(self.W_out_ns.t(), g)
        grad_out = torch.ger(center_vector, g).t()

        # update weights
        self.W_emb[center_emb_index] -= self.learning_rate * grad_emb
        self.W_out_ns -= self.learning_rate * grad_out


    return loss / len(context_emb_indices)
# switch self.func to this func for the model
self.func = skipgram_ns_modified





def n_most_frequent(steps_list):
    # hacky code to minimmise code editing
    model = self

    print()
    for n in steps_list:
        print("The top", n, "most frequent words: ")
        print("occurrence count:\t", sum([model.occurrence_dict[model.ind2word[i]] for i in range(0, n)]))
        print("length of corpus:\t", len(model.corpus))
        print("percentage:\t\t %.3f" % (sum([model.occurrence_dict[model.ind2word[i]] for i in range(0, n)]) / len(model.corpus) * 100), "%")
        print()

    print("total vocabulary size:", len(model.occurrence_dict))
    print()




def main():
    model = self
    model.train(iteration=400000, output_filename="w2v_analyze_model", debug=True, verbose=False)
    print()



    # view embedding similarities
    # the model learns some words which are similar for approximately 2~3% of the words
    
    # one epoch can be said to be approximately 72K iterations,
    # if we consider the traversal of the full vocabulary to be one epoch

    # however, since stochastic gradient descent is used,
    # only 60% of the words will get "touched" by the model as the target word approximately
    # (i.e. have their W_emb updated in a particular epoch)

    # a high learning rate is therefore helpful in speeding up the learning process

    # loss is unlikely to go down until enough words have been learnt by the model to make a difference
    # its invariance is especially notable when using negative sampling
    # since the loss is calculated w.r.t. only a few samples (15 + target)

    if True:
        #word_list = ["anarchist", "revolution", "although", "william", "diggers", "the", "by", "of", "one", "eight"]

        if False: # 28s for 300 words (incl. model loading)
            for i, word in enumerate(model.word2ind.keys()):
                if i < 4000:
                    continue
                w = model.find_similar_fast(word, 5)
                for pair in w[1:]:
                    if pair[1] > 0.3:
                        print(i, w)
                        break
                if i % 100 == 0:
                    print("i:\t", i)
        
    
        # speed test comparing find_similar vs find_similar_faster
        if False: # 45s for 300 words (incl. model loading)
            for i, word in enumerate(model.word2ind.keys()):
                w = model.find_similar(word, 5)
                for pair in w[1:]:
                    if pair[1] > 0.4:
                        print(w)
                        break
                if i == 300:
                    print("i:\t", i)
                    break


    



    # define question words
    qn_words = ["brother", "sister", "grandson", "granddaughter",
                "apparent", "apparently", "rapid", "rapidly",
                "possibly", "impossibly", "ethical", "unethical",
                "great", "greater", "tough", "tougher",
                "easy", "easiest", "lucky", "luckiest",
                "think", "thinking", "read", "reading",
                "walking", "walked", "swimming", "swam",
                "mouse", "mice", "dollar", "dollars",
                "work", "works", "speak", "speaks"]
    
    # analogical reasoning task
    # note that none of the original question words have been "learnt" by the model
    # (i.e. there is another word which has similarity > 0.4)
    if True:

        # this works (as of 120K epochs; skipgram-negative-sampling-with-subsampling-with-lr-of-1)
        #qn_words = ["franc", "francs", "midwife", "midwives"]
        
        
        # perform analogical reasoning task
        # find similar words
        print("Checking question words for which the model has \"learnt\" something...")
        for word in qn_words:
            word_list = model.find_similar_fast(word, 5)
            for word in word_list[1:]:
                # the model has yet to learn any of the words even after 280K iterations
                # "lucky" does not show up even with a threshold of 0.2
                if word[1] > 0.3:
                    print(word_list)
                    print()
                    break
        print("Checking complete.")
        print()

    if False:
        for i in range(len(qn_words) / 4):
            answer = model.deduce(qn_words[i * 4], qn_words[i * 4 + 1], qn_words[i * 4 + 3])
            print("question:", qn_words[i * 4], "-", qn_words[i * 4 + 1], "+", qn_words[i * 4 + 3])
            print("answer:", answer)  # TODO: better format for output
            print()
            answer = model.deduce(qn_words[i * 4 + 1], qn_words[i * 4], qn_words[i * 4 + 2])
            print("question:", qn_words[i * 4 + 1], "-", qn_words[i * 4], "+", qn_words[i * 4 + 2])
            print("answer:", answer)  # TODO: better format for output
            print()
            answer = model.deduce(qn_words[i * 4 + 2], qn_words[i * 4 + 3], qn_words[i * 4 + 1])
            print("question:", qn_words[i * 4 + 2], "-", qn_words[i * 4 + 3], "+", qn_words[i * 4 + 1])
            print("answer:", answer)  # TODO: better format for output
            print()
            answer = model.deduce(qn_words[i * 4 + 3], qn_words[i * 4 + 2], qn_words[i * 4])
            print("question:", qn_words[i * 4 + 3], "-", qn_words[i * 4 + 2], "+", qn_words[i * 4])
            print("answer:", answer)  # TODO: better format for output
            print()
        print()





    # check freq rank of qn words
    if False:
        qn_word_index_list = []
        for word in qn_words:
            # show how frequent the qn_words are
            qn_word_index_list.append([word, model.word2ind[word]])
        sorted_list = sorted(qn_word_index_list, key=lambda x: x[1])
        for word in sorted_list:
            print("%15s: %d" % (word[0], word[1]))
        print()
        print()





    # find words similar to a certain set of words
    if False:
        for i in range(0, 50):

            # select words by frequency or by position in corpus
            word = model.ind2word[i]
            #word = model.corpus[i]

            # note: word_list contains tuples
            word_list = model.find_similar_fast(word, 5)
            print("words most similar to:", word)
            print(word_list)
            print()
        print()
        # print(model.corpus[:100]) # optionally print part of the corpus for reference




    # Assignment 4:
    # analyse the proportion of the n most frequent words
    # note that the most frequent 50 words take up more than 40% of the corpus,
    # which is why subsampling/removal of common words may be strongly required
    # for proper training.
    
    # Assignment 5:
    # when the model is subsampled, the proportion of the most common words decreases significantly
    if False:
        f_list = [10, 50, 100, 300, 500, 2000, 3000, 5000, 10000]
        n_most_frequent(f_list)

        #print()
        #print("subsampling... (1e-3)")
        #model.subsample_corpus(1e-3)
        #print()
        
        #n_most_frequent(f_list)

        print()
        print("subsampling... (1e-4)")
        model.subsample_corpus(1e-4)
        print()
        
        n_most_frequent(f_list)

        print()
        print("subsampling... (1e-5)")
        model.subsample_corpus(1e-5)
        print()
        
        n_most_frequent(f_list)

if __name__ == "__main__":
    main()