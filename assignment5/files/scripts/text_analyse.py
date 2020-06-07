from collections import Counter

# obtain corpus and word count
corpus = open('text8', mode='r')
corpus = corpus.readlines()[0]#[:10000]
corpus = corpus.split()
corpus_count = Counter(corpus)

print("corpus length: " + str(len(corpus)))
print("number of words : " + str(len(corpus_count)))
print()
print("20 most common words: ", corpus_count.most_common()[:20])
print()
print("some least common words: ", corpus_count.most_common()[-20:])
print()



# calculate some percentages
threshold = 4 # a "hyperparameter" that affects how many words are removed


# calculate the number of occurrences of words with less than a certain number of occurrences
n_times = []
for i in range(threshold * 5):
    n_times.append([word for word in corpus_count if corpus_count[word] == i])

for i in range(1, threshold * 2 + 1):
    print(str(i) + ": " + str(len(n_times[i])))

n_words = 0
n_occ_rem = 0
for i in range(threshold):
    n_words += len(n_times[i])
    n_occ_rem += len(n_times[i]) * i

print()
print("threshold: ", threshold)
print("number of words removed: ", n_words)
print("proportion of words removed : ",  n_words / len(corpus_count))
print("number of instances removed: ", n_occ_rem)
print("proportion of corpus removed: ",  n_occ_rem / len(corpus))
print()
print("these words won't be removed: ", n_times[threshold + 1][:10])
print()
print()
# note that almost half the words only appear once in the corpus
# since the embeddings of extremely rare words will not be meaningful,
# they will be removed.


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

qn_words_occ = {word : corpus_count[word] for word in qn_words}
qn_words_sort = sorted(qn_words_occ, key=qn_words_occ.get, reverse=True)

for i in range(len(qn_words)):
    print(qn_words_sort[i], ": ", corpus_count[qn_words_sort[i]])
    #if i == 10:
    #    print("(anything beyond this point is unlikely to work.)")

# note that words such as "impossibly" and "luckiest" are extremely rare in the corpus, 
# albeit not removed with a threshold of 4.

# 10e4 / (1.5 * 10e7) < 7 * 10e-4 => there's a less than 0.07% chance that even the most common word
# in the list of question words will be picked as the center word at each iteration.