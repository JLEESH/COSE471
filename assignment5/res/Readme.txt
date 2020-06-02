Word2vec2
(Hierarchical Softmax, Negative Sampling, Subsampling)

Implement skip-gram and CBOW models in word2vec.py
If you run "word2vec.py", you can train and test your models.
--------------------------------------------------------------------------------------------------
How to run

python word2vec.py [mode] [ns] [subsampling] [partition]

mode : "SG" for skipgram, "CBOW" for CBOW
ns : "0" for hierarchical softmax, the other numbers would be the number of negative samples
subsampling : "Y" for using subsampling, "N" for not
partition : "part" if you want to train on a part of corpus (fast training but worse performance), 
             "full" if you want to train on full corpus (better performance but very slow training)

Examples) 
python word2vec.py SG 0 Y part
python word2vec.py CBOW 5 N part
python word2vec.py SG N 3 full
python word2vec.py CBOW Y 5 full

You should adjust the other hyperparameters in the code file manually.
--------------------------------------------------------------------------------------------------