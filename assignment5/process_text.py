from collections import Counter
import pickle

print("loading corpus...")
corpus = open('text8', mode='r')
corpus = corpus.readlines()[0]#[:10000]
corpus = corpus.split()
corpus_count = Counter(corpus)

threshold = 4
corpus_count_processed = {}
print("creating occurrence dictionary...")
for word in corpus_count:
    if corpus_count[word] > threshold:
        corpus_count_processed[word] = corpus_count[word]

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
    #if i == 10:
    #    print("(anything beyond this point is unlikely to work.)")

# note that words such as "impossibly" and "luckiest" are extremely rare in the corpus, 
# albeit not removed with a threshold of 4.

# 10e4 / (1.5 * 10e7) < 7 * 10e-4 => there's a less than 0.07% chance that even the most common word
# in the list of question words will be picked as the center word at each iteration.




# save variables using pickle
with open("w2v_vars", mode="wb") as f:
    pickle.dump(corpus_processed, f)
    pickle.dump(corpus_count_processed, f)
    pickle.dump(word2ind, f)
    pickle.dump(ind2word, f)