import numpy as np
import torch
import torch.nn as nn

def sortSecond(elt):
    return elt[1]

def sim(u,v):
    return np.dot(u,v)/(np.sqrt(np.dot(u,u)) * np.sqrt(np.dot(v,v)))

def most_similar(word, k , word_to_index, embedding_vocab):
    indx = word_to_index[word]
    list = []
    
    for i in range(len(word_to_index)):
        if i == indx: continue
        temp = []
        temp.append(i)
        A= sim(embedding_vocab[indx], embedding_vocab[i])
        temp.append(A)
        list.append(temp)
    list.sort(key=sortSecond, reverse=True)
    return list[0:k]







def main():
    # embedding vector http://u.cs.biu.ac.il/~89-687/ass2/wordVectors.txt
    # corresponding word  http://u.cs.biu.ac.il/~89-687/ass2/vocab.txt
    
    path_vocab = "q2\\vocab.txt"
    path_wordVector = "q2\\wordVectors.txt"
    
    
    
    target_vocab = []
    with open(path_vocab) as f:
        for line in f:
            target_vocab.append(line.strip())

    embedding_vocab = np.loadtxt(path_wordVector)

    word_to_index = {target_vocab[i]:i for i in range(len(target_vocab))}
    indx_to_word = {i: target_vocab[i] for i in range(len(target_vocab))}

# print(embedding_vocab[word_to_index[target_vocab[0]]])
    words = ['dog', 'england', 'john', 'explode', 'office']
    for w in words:
        similar_indeces = most_similar(w,5,word_to_index,embedding_vocab)
        similar_words = [(indx_to_word[pair[0]],'ditance=' +str(pair[1])) for pair in similar_indeces]

        print("word: '" + w +"':\n" + str(similar_words) + "\n")



main()
