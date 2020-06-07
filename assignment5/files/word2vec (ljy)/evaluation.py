import torch

emb, word2ind, ind2word = torch.load('./결과/SG_NS.pt', map_location=torch.device('cpu'))

emb_lengths = (emb * emb).sum(1) ** 0.5
emb_normalised = emb / emb_lengths[:, None]

def similar_words(word):
    word_normalised = word / torch.norm(word)
    similarity = (emb_normalised * word_normalised).sum(1)

    values, indices = similarity.topk(10)

    word_lst = []
    for index, value in zip(indices, values):
        word_lst.append((ind2word[index.item()], value.item()))

    return word_lst

word1 = ["brother", "sister", "grandson", "granddaughter"]
word2 = ["apparent", "apparently", "rapid", "rapidly"]
word3 = ["possibly", "impossibly", "ethical", "unethical"]
word4 = ["great", "greater", "tough", "tougher"]
word5 = ["easy", "easiest", "lucky", "luckiest"]
word6 = ["think", "thinking", "read", "reading"]
word7 = ["walking", "walked", "swimming", "swam"]
word8 = ["mouse", "mice", "dollar", "dollars"]
word9 = ["work", "works", "speak", "speaks"]

words = [word1, word2, word3, word4, word5, word6, word7, word8, word9]

for word in words:
    for w in word:
        print("similar to word:", w)
        print(similar_words(emb[word2ind[w]]))
        print()
for word in words:
    test_case1 = emb[word2ind[word[2]]] - emb[word2ind[word[3]]] + emb[word2ind[word[1]]]
    test_case2 = emb[word2ind[word[3]]] - emb[word2ind[word[2]]] + emb[word2ind[word[0]]]
    test_case3 = emb[word2ind[word[0]]] - emb[word2ind[word[1]]] + emb[word2ind[word[3]]]
    test_case4 = emb[word2ind[word[1]]] - emb[word2ind[word[0]]] + emb[word2ind[word[2]]]
    test_cases = [test_case1, test_case2, test_case3, test_case4]
    for idx, test_case in enumerate(test_cases):
        guess = similar_words(test_case)
        print("answer:", word[idx])
        print(guess)
    print()