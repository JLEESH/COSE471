import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import random
import argparse
import pickle



'''
process_text(filename, pickle_filename)


Preprocesses corpus given in filename to generate and save certain variables into pickle_filename.
'''
def process_text(filename, pickle_filename):
    ## dependencies
    #from collections import Counter
    #import pickle

    print("loading corpus...")
    corpus = open(filename, mode='r')
    corpus = corpus.readlines()[0]#[:10000]
    corpus = corpus.split()
    corpus_count = Counter(corpus)

    threshold = 4
    corpus_count_processed = {}
    print("creating occurrence dictionary...")
    for word in corpus_count:
        if corpus_count[word] > threshold:
            corpus_count_processed[word] = corpus_count[word]

    # create a processed corpus that contains all the words in the dictionary
    print("processing corpus...")
    corpus_processed = [word for word in corpus if word in corpus_count_processed]
    
    word2ind = dict()
    ind2word = dict()
    print("creating word2ind and ind2word...")
    for index, word in enumerate(corpus_count.most_common()):
        word2ind[word[0]] = index
        ind2word[index] = word[0]

    bindex = 0
    n_words = 20
    print("taking a peek at word2ind...")
    print()
    for i in range(bindex, bindex + n_words):
        print(ind2word[i], ":", (word2ind[ind2word[i]]))

    print()
    print("original corpus length: ", len(corpus))
    print("reduced corpus length: ", len(corpus_processed))
    print("original vocabulary size: ", len(corpus_count))
    print("reduced vocabulary size: ", len(corpus_count_processed))
    print()

    print("checking that none of the question words are being removed...")
    print()

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

    qn_words_occ = {word : corpus_count_processed[word] for word in qn_words}
    qn_words_sort = sorted(qn_words_occ, key=qn_words_occ.get, reverse=True)

    for i in range(len(qn_words)):
        print(qn_words_sort[i], "appears", corpus_count_processed[qn_words_sort[i]], "times.")
    print()

    # note that words such as "impossibly" and "luckiest" are extremely rare in the corpus, 
    # albeit not removed with a threshold of 4.

    # save variables using pickle
    with open(pickle_filename, mode="wb") as f:
        pickle.dump(corpus_processed, f)
        pickle.dump(corpus_count_processed, f)
        pickle.dump(word2ind, f)
        pickle.dump(ind2word, f)


# TODO: implement hierarchical softmax and negative sampling
# TODO: implement subsampling and support other affected operations (saving, loading etc.)
class Word2Vec():

    '''
    Word2Vec(architecture="skipgram", mode="negative_sampling", subsample=True,
                dimension=300, max_context_dist=4, learning_rate=1e-3,
                pickle_filename=None, load_model=False, model_filename=None):

    Creates a word2vec model with the given parameters.

    Raises ValueError if an invalid architecture name is provided (i.e. other than "skipgram" or "cbow").
    Raises ValueError if an invalid mode is provided.
    '''
    def __init__(self, architecture="skipgram", mode="negative_sampling", subsample=True,
                    dimension=300, max_context_dist=4, learning_rate=1e-3,
                    pickle_filename=None, load_model=False, model_filename=None):


        # determine architecture
        if architecture == "skipgram":
            self.architecture = "skipgram"
            self.func = self.skipgram
        elif architecture == "cbow":
            self.architecture = "cbow"
            self.func = self.cbow
        else:
            raise ValueError("Architecture not recognised.")

        # define certain hyperparameters
        self.learning_rate = learning_rate
        self.window = max_context_dist
        self.dimension = dimension


        # select mode to use
        if mode == "negative_sampling":
            raise NotImplementedError("Negative sampling not implemented.")
        elif mode == "hierarchical_softmax":
            raise NotImplementedError("Hierarchical softmax not implemented.")
        elif mode == "none":
            pass
        else:
            raise ValueError("Invalid word2vec mode.")

        self.mode = mode
        self.subsample = subsample


        # perform subsampling
        # TODO: implement subsampling operation and support loading and saving models trained on a subsampled corpus
        if self.subsample:

            # set seed to use for reproducibility
            self.seed = random.randrange(0, 10, 1) # range of 0 to 9 (for now)
            random.seed(self.seed)
        



        # load preprocessed corpus and relevant dictionaries
        if pickle_filename is not None:
            self.load_w2v_vars(pickle_filename)
            self.corpus_size = len(self.corpus)
            self.vocabulary_size = len(self.occurrence_dict)
        else:
            self._process_corpus("text8")


        # initialise or load embedding weights
        if load_model:
            if model_filename is None:
                raise ValueError("model_filename not provided.")
            self.load_model(model_filename)

        else:
            self.W_emb = torch.randn(self.vocabulary_size, self.dimension) / dimension**0.5
            self.W_out = torch.randn(self.vocabulary_size, self.dimension) / dimension**0.5



    '''
    load_model(filename)


    Loads certain parameters saved during training.
    '''
    def load_model(self, filename):

        print("loading model...")
        W_dict = torch.load(filename)
        self.W_emb = W_dict["W_emb"]
        self.W_out = W_dict["W_out"]

        if self.subsample:
            self.seed = W_dict["seed"]
            print("subsampling seed:", self.seed)
        
        if self.mode != W_dict["mode"]:
            print("WARNING: mode mismatch detected.")


    '''
    load_w2v_vars(filename)


    Loads corpus, occurrence_dict, word2ind and ind2word.
    
    Note: requires a pickle file generated by the appropriate preprocessor.
    '''
    def load_w2v_vars(self, filename):
        with open(filename, "rb") as f:
            self.corpus = pickle.load(f)
            self.occurrence_dict = pickle.load(f)
            self.word2ind = pickle.load(f)
            self.ind2word = pickle.load(f)



    '''
    _process_corpus(self, filename)


    Internal method to process the corpus. Creates a file named w2v_vars_process_corpus.
    '''
    def _process_corpus(self, filename):
        # TODO: streamline corpus preprocessing(low priority)
        p_filename = "w2v_vars_process_corpus"
        
        # TODO: alternative process_text method with subsampling
        process_text(filename, p_filename)


        self.load_w2v_vars(p_filename)
        self.corpus_size = len(self.corpus)
        self.vocabulary_size = len(self.occurrence_dict)


    '''
    save_weights(filename)


    Save weights and, when required, other variables.
    '''
    def save_weights(self, filename):
        if self.verbose:
            print()
            print("saving model to file...")
        
        W_dict = {"W_emb" : self.W_emb, "W_out" : self.W_out}
        
        if self.subsample:
            W_dict["seed"] = self.seed
        
        if self.mode != "none":
            W_dict["mode"] = self.mode
        
        torch.save(W_dict, filename)

        if self.verbose:
            print("model saved to file successfully.")
            print()

    '''
    getContextIndices(center_index)


    Obtains context indices based on the center index by searching through the corpus
    depending on the maximum window size.

    Returns a list of context indices.

    Warning: the size of the list varies according to the availability of the context
    (i.e. the full window size may not be reached for words close to the beginning or
    the end of the corpus)
    '''
    def getContextIndices(self, center_index):
        low = max(0, center_index - self.window)
        high = min(center_index + self.window, len(self.corpus) - 1)
        context_indices = [a for a in range(low, high + 1)]
        context_indices.remove(center_index)
        return context_indices



    '''
    forward(center_index, context_indices)


    Returns loss.
    '''
    def forward(self, center_index, context_indices):
        return self.func(center_index, context_indices)



    '''
    predict(word)


    Makes predictions based on the word input.
    Warning: not implemented.
    '''
    def predict(self, word):
        raise NotImplementedError("predict() not implemented.")
    


    '''
    find_similar(word, n_output=5)


    Finds a number of words that are the most similar to a given word.
    '''
    def find_similar(self, word, n_output=5):
        # TODO: use Pytorch cosine similarity instead


        # obtain lengths of embedding vectors
        
        #  (W_i . W_i) ^ 0.5 = (||W_i|| ^ 2) ^ 0.5 = ||W_i|| for each i
        #  (elementwise product, sum each row, take square root)
        W_emb_lengths = (self.W_emb * self.W_emb).sum(1) ** 0.5

        #  W / ||W||
        W_emb_normalised = self.W_emb / W_emb_lengths[:, None]

        #  W_n . d_word
        similarity = (W_emb_normalised * W_emb_normalised[self.word2ind[word]]).sum(1)


        # find vectors that are the most similar
        values, indices = similarity.topk(n_output)

        # generate word_list
        word_list = []
        for index, value in zip(indices, values):
            word_list.append((self.ind2word[index.item()], value.item()))

        # set breakpoint here to take a look at word_list and other local variables
        return word_list


    '''
    deduce(word_original, word_root, word_question)


    Performs an analogical reasoning task based on the given words.
    i.e. finds the words closest to the following word embedding vector: 

        word_original - word_root + word_question

    Returns a list of the 5 most similar words along with their similarity values.
    '''
    def deduce(self, word_original, word_root, word_question):
        # TODO: use Pytorch cosine similarity instead

        W_emb = self.W_emb
        word2ind = self.word2ind

        # obtain representation
        word_answer = W_emb[word2ind[word_original]] - W_emb[word2ind[word_root]] + W_emb[word2ind[word_question]]

        # normalise representation
        word_answer = word_answer / torch.norm(word_answer)

        # obtain lengths of embedding vectors 
        W_emb_lengths = (self.W_emb * self.W_emb).sum(1) ** 0.5
        W_emb_normalised = self.W_emb / W_emb_lengths[:, None]
        similarity = (W_emb_normalised * word_answer).sum(1)

        # find 10 words most similar to the representation
        values, indices = similarity.topk(10)

        word_list = []
        for index, value in zip(indices, values):
            word_list.append((self.ind2word[index.item()], value.item()))

        return word_list



    '''
    train(iteration=50000, output_filename=None, debug=False, verbose=False, train_partial=False)


    Trains the model according to the given parameters.
    '''
    def train(self, iteration=50000, output_filename=None, debug=False, verbose=False, train_partial=False):
        
        # check if the user wants to debug
        self.debug = debug
        self.verbose = verbose
        self.train_partial = train_partial

        # set progress checking and weight saving iterations
        if self.architecture == "skipgram":
            self.progress_check_iteration = 500
            self.weight_save_iteration = 5000

        elif self.architecture == "cbow":
            self.progress_check_iteration = 1000
            self.weight_save_iteration = 10000
        

        # check if the model should keep training
        inf_train = False
        if iteration < 0:
            inf_train = True

        # initialise loss list
        loss_list = []

        # begin training
        i = 0
        while i < iteration or inf_train:
            i += 1

            # obtain random center word index
            center_index = random.randint(0, self.corpus_size - 1)

            # check if the user wants to train on a partial corpus
            if train_partial:
                center_index = center_index % 1000

            # obtain context_indices
            context_indices = self.getContextIndices(center_index)

            # train using self.func
            loss = self(center_index, context_indices)
            loss_list.append(loss)


            ############ DEBUG: test getContextIndices() #############
            if self.debug and False:
                center = self.corpus[center_index]
                context = [self.corpus[index] for index in context_indices]
                if verbose:
                    print(center)
                    print(context)
            ######################## END DEBUG ########################


            # average loss calculation
            if i % self.progress_check_iteration == 0:
                loss_avg = sum(loss_list) / len(loss_list)
                print("iteration:", i)
                print("loss:", loss_avg)
                loss_list = []
            

            # save weights to file automatically after a certain number of iterations
            if i % self.weight_save_iteration == 0:
                self.save_weights(output_filename)


        # save weights to file after training is complete
        self.save_weights(output_filename)

    

    '''
    skipgram(center_index, context_indices)


    Performs skipgram training. Returns loss as a primitive.
    '''
    def skipgram(self, center_index, context_indices):
        if self.mode == "negative_sampling":
            raise NotImplementedError("skipgram() in negative sampling mode not implemented.")

        if self.mode == "hierarchical_softmax":
            raise NotImplementedError("skipgram() in hierarchical softmax mode not implemented.")
        
        elif self.mode == "none":
            
            # obtain embedding indices
            center_emb_index = self.word2ind[self.corpus[center_index]]
            context_emb_indices = [self.word2ind[self.corpus[index]] for index in context_indices]
            
            # feed forward
            center_vector = self.W_emb[center_emb_index]
            output = torch.mv(self.W_out, center_vector)
            softmax = F.softmax(output, 0)

            # calculate loss and gradients
            for index in context_emb_indices:
                g = softmax.clone()
                g[index] -= 1
                loss = -1 * F.log_softmax(output, 0)[index]
                grad_emb = torch.mv(self.W_out.t(), g)
                grad_out = torch.ger(center_vector, g).t()

                # update weights
                self.W_emb[center_emb_index] -= self.learning_rate * grad_emb
                self.W_out -= self.learning_rate * grad_out

            return loss.item()
    


    '''
    cbow(center_index, context_indices)


    Performs CBOW training. Returns loss as a primitive.
    '''
    def cbow(self, center_index, context_indices):
        if self.mode == "negative_sampling":
            raise NotImplementedError("skipgram() in negative sampling mode not implemented.")

        if self.mode == "hierarchical_softmax":
            raise NotImplementedError("skipgram() in hierarchical softmax mode not implemented.")
        
        elif self.mode == "none":

            # obtain embedding indices
            center_emb_index = self.word2ind[self.corpus[center_index]]
            context_emb_indices = [self.word2ind[self.corpus[index]] for index in context_indices]
            
            # feed forward
            context_vector = torch.zeros_like(self.W_emb[center_emb_index])
            for index in context_emb_indices:
                context_vector += self.W_emb[index]
            output = torch.mv(self.W_out, context_vector)
            softmax = F.softmax(output, 0)

            # calculate loss and gradients
            softmax[center_emb_index] -= 1
            loss = -1 * F.log_softmax(output, 0)[center_emb_index]
            grad_emb = torch.mv(self.W_out.t(), softmax)
            grad_out = torch.ger(context_vector, softmax).t()

            # update weights
            self.W_emb[context_emb_indices] -= self.learning_rate * grad_emb
            self.W_out -= self.learning_rate * grad_out

            return loss.item()



def main():
    # TODO: clean up code for submission/out-of-the-box testing

    # parse commandline arguments
    parser = argparse.ArgumentParser("word2vec")
    parser.add_argument("-s", "--skip-task", dest="task", default=True, action="store_false")
    parser.add_argument("-n", "--no-train", dest="train", default=True, action="store_false")
    parser.add_argument("-d", "--debug", dest="debug", default=False, action="store_true")
    parser.add_argument("-p", "--train-partial", dest="train_partial", default=False, action="store_true")
    parser.add_argument("-i", "--inf-train", dest="inf_train", default=False, action="store_true")
    parser.add_argument("-l", "--load-model", dest="load_model", default=False, action="store_true")
    parser.add_argument("-r", "--learning-rate", dest="learning_rate", default=5e-3, metavar="lr", type=float)
    parser.add_argument("-e", "--iterations", dest="iterations", default=10000, metavar="it", type=int)

    args = parser.parse_args()
    perform_task = args.task
    perform_training = args.train
    debug = args.debug
    verbose = args.debug # assume verbose is true when debug is true
    train_partial = args.train_partial
    inf_train = args.inf_train
    load_model = args.load_model
    learning_rate = args.learning_rate
    iterations = args.iterations


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

    # define some "easier" test questions with more common words
    qn_words_test = ["is", "was", "has", "had",
                "is", "are", "has", "have",
                "is", "was", "are", "were",
                #"were", "was", "have", "has",
                "he", "his", "she", "hers",
                #"their", "they", "his", "he",
                #"their", "they", "her", "she",
                "day", "days", "year", "years",
                "days", "day", "books", "book",
                "french", "france", "german", "germany",
                "one", "two", "first", "second"]
    

    # create model
    # NOTE: the text8 file must be in the same directory to generate the pickle file.

    if load_model:
        model = Word2Vec("cbow", mode="none", learning_rate=learning_rate, load_model=load_model, model_filename="w2v_model_with_embeddings", pickle_filename="w2v_vars")
    else:
        model = Word2Vec("cbow", mode="none", learning_rate=learning_rate, load_model=load_model, model_filename="w2v_model_with_embeddings", pickle_filename=None) # to make it explicit that we are not loading a pickle file
    
    # train model (takes a long time)
    if perform_training:
        print("commencing training...")
        if inf_train:
            model.train(-1, output_filename="w2v_model_with_embeddings", debug=debug, verbose=verbose, train_partial=train_partial)
        else:
            model.train(iterations, output_filename="w2v_model_with_embeddings", debug=debug, verbose=verbose, train_partial=train_partial)
            ## notify user when training ends
            #import winsound
            #winsound.Beep(2500, 1000)

    if perform_task:
        # find similar words
        for word in qn_words:
            word_list = model.find_similar(word, 5)
            print("words most similar to:", word)
            print(word_list)
            print()
        print()
        
        # perform analogical reasoning task
        for i in range(0, 9):
            answer = model.deduce(qn_words[i * 4], qn_words[i * 4 + 1], qn_words[i * 4 + 3])
            print("question:", qn_words[i * 4], "-", qn_words[i * 4 + 1], "+", qn_words[i * 4 + 3])
            print("answer:", answer) # TODO: better format for output
            print()
            answer = model.deduce(qn_words[i * 4 + 1], qn_words[i * 4], qn_words[i * 4 + 2])
            print("question:", qn_words[i * 4 + 1], "-", qn_words[i * 4], "+", qn_words[i * 4 + 2])
            print("answer:", answer) # TODO: better format for output
            print()
            answer = model.deduce(qn_words[i * 4 + 2], qn_words[i * 4 + 3], qn_words[i * 4 + 1])
            print("question:", qn_words[i * 4 + 2], "-", qn_words[i * 4 + 3], "+", qn_words[i * 4 + 1])
            print("answer:", answer) # TODO: better format for output
            print()
            answer = model.deduce(qn_words[i * 4 + 3], qn_words[i * 4 + 2], qn_words[i * 4])
            print("question:", qn_words[i * 4 + 3], "-", qn_words[i * 4 + 2], "+", qn_words[i * 4])
            print("answer:", answer) # TODO: better format for output
            print()
        print()
    
    
    ## make predictions
    #model.predict("hello")


    '''
    ###### DEBUG #####

    The following section produces output for debugging.
    TODO: use commandline arguments to streamline the debugging process?
    '''

    if debug:# and False:

        # debug: find words similar to a certain set of words
        for i in range(20):
            
            # select words by frequency or by position in corpus
            word = model.ind2word[i]
            #word = model.corpus[i]

            word_list = model.find_similar(word, 15) # note: word_list contains tuples
            print("words most similar to:", word)
            print(word_list)
            print()
        print()
        #print(model.corpus[:100]) # optionally print part of the corpus for reference


        # perform analogical reasoning task on easier questions.
        for i in range(0, 5):
            answer = model.deduce(qn_words_test[i * 4], qn_words_test[i * 4 + 1], qn_words_test[i * 4 + 2])
            print("answer:", answer) # TODO: better format for output
            answer = model.deduce(qn_words_test[i * 4 + 1], qn_words_test[i * 4], qn_words_test[i * 4 + 3])
            print("answer:", answer) # TODO: better format for output
            answer = model.deduce(qn_words_test[i * 4 + 2], qn_words_test[i * 4 + 3], qn_words_test[i * 4])
            print("answer:", answer) # TODO: better format for output
            answer = model.deduce(qn_words_test[i * 4 + 3], qn_words_test[i * 4 + 2], qn_words_test[i * 4 + 1])
            print("answer:", answer) # TODO: better format for output
            print()
        print()


    # analyse the proportion of the n most frequent words
    # note that the most frequent 50 words take up more than 40% of the corpus,
    # which is why subsampling/removal of common words may be strongly required
    # for proper training.
    if debug:# and False:
        print()
        for n in [10, 50, 100, 300, 500, 2000, 3000]:
            print("The top", n, "most frequent words: ")
            print("occurrence count:\t", sum([model.occurrence_dict[model.ind2word[i]] for i in range(0, n)]))
            print("length of corpus:\t", len(model.corpus))
            print("percentage:\t\t %.3f" % (sum([model.occurrence_dict[model.ind2word[i]] for i in range(0, n)]) / len(model.corpus) * 100), "%")
            print()

        print("total vocabulary size:", len(model.occurrence_dict))
        print()
    print()
    

    # take a look at the word embeddings
    if debug and False:
        for i in range(20):
            print(model.ind2word[i], "\t:\t", model.W_emb[i, :5])
    

    # misc. debug
    if debug and False:
        print("learning rate:", learning_rate)



if __name__ == '__main__':
    main()