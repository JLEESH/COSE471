import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import random
import argparse
import pickle
from w2v import Word2Vec


model_filename = "./models/w2v_model_skipgram_ns_subsample_lr1 (52K) (proper subsampling)"
model_filename = "w2v_model_skipgram_ns_subsample_lr1"
self = Word2Vec(architecture="skipgram", mode="negative_sampling", subsample=True,
                model_filename=model_filename, pickle_filename="w2v_vars_process_corpus",
                load_model=True, debug=True, learning_rate=1)

'''
skipgram_ns(center_index, context_indices)


Performs skipgram training with negative sampling. Returns loss as a primitive.
'''
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

self.func = skipgram_ns_modified


def main():
    model = self
    #model.train(iteration=49000, output_filename="w2v_analyze_model", debug=True, verbose=True, train_partial=True)
    
    if True:
        #word_list = ["anarchist", "revolution", "although", "william", "diggers", "the", "by", "of", "one", "eight"]

        if True: # 28s for 300 words (incl. model loading)
            for i, word in enumerate(model.word2ind.keys()):
                w = model.find_similar_fast(word, 5)
                for pair in w[1:]:
                    if pair[1] > 0.4:
                        print(w)
                        break
                if i % 100 == 0:
                    print("i:\t", i)
        
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

    
    if False:
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
        
        #qn_words = ["defines", "define", "makes", "store"]
        
        # perform analogical reasoning task
            # find similar words
        for word in qn_words:
            word_list = model.find_similar_fast(word, 5)
            for word in word_list[1:]:
                if word[1] > 0.4:
                    print("words most similar to:", word)
                    print(word_list)
                    print()
        print()
        for i in range(0, 1):
            answer = model.deduce(qn_words[i * 4], qn_words[i * 4 + 1], qn_words[i * 4 + 3])
            print("question:", qn_words[i * 4], "-", qn_words[i * 4 + 1], "+", qn_words[i * 4 + 3])
            print("answer:", answer)  # TODO: better format for output
            print()
            #answer = model.deduce(qn_words[i * 4 + 1], qn_words[i * 4], qn_words[i * 4 + 2])
            #print("question:", qn_words[i * 4 + 1], "-", qn_words[i * 4], "+", qn_words[i * 4 + 2])
            #print("answer:", answer)  # TODO: better format for output
            #print()
            #answer = model.deduce(qn_words[i * 4 + 2], qn_words[i * 4 + 3], qn_words[i * 4 + 1])
            #print("question:", qn_words[i * 4 + 2], "-", qn_words[i * 4 + 3], "+", qn_words[i * 4 + 1])
            #print("answer:", answer)  # TODO: better format for output
            #print()
            #answer = model.deduce(qn_words[i * 4 + 3], qn_words[i * 4 + 2], qn_words[i * 4])
            #print("question:", qn_words[i * 4 + 3], "-", qn_words[i * 4 + 2], "+", qn_words[i * 4])
            #print("answer:", answer)  # TODO: better format for output
            #print()
        print()

if __name__ == "__main__":
    main()