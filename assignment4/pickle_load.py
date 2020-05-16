import pickle
from collections import Counter

with open("w2v_vars", mode="rb") as f:
    corpus_processed = pickle.load(f)
    corpus_count_processed = pickle.load(f)
    word2ind = pickle.load(f)
    ind2word = pickle.load(f)


print(len(corpus_processed), len(corpus_count_processed))
print(ind2word[100], word2ind[ind2word[100]])