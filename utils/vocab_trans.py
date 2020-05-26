import sys
sys.path.append("..")
from utils.data_slice import *
from gensim.models import word2vec
import jieba
import numpy as np

def vocab_encoder_gen(filepath, modelpath):
    print(filepath)
    texts, labels = load_texts_and_labels(filepath)
    one_hot_labels = []
    x_datas = []
    for i,line in enumerate(texts):
        #parts = line.split('\t',1)
        if(len(line.strip()) == 0):
            continue

        parts= list(jieba.cut(line, cut_all=False))
        x_datas.append(parts)

    x_text = x_datas

    max_document_length = max([len(x) for x in x_text])
    print ('len(x) = ',len(x_text))
    print(' max_document_length = ' , max_document_length)
    model = word2vec.Word2Vec.load(modelpath)   
    vects = model.wv.vocab
    vocabs = list(vects.keys())
    vec = []
    for i in vocabs:
        vec.append(model[i])
    vocab_size = len(vocabs)
    x = [[0] * (max_document_length+1) for i in range(len(x_text))]
    #print(x[7851][1])    
    output_vocab = open('vocab.txt', 'w', encoding='utf-8')
    for i in vocabs:
        output_vocab.write(i+'\n')
    output_vocab.close()


    output = open('trans.txt', 'w', encoding='utf-8')
    for i,line in enumerate(x_text):
        if i % 1000 == 0:
            print("processing %d line"%(i))
        for j in range(len(line)):
            if line[j] in vocabs:
                x[i][j] = vocabs.index(line[j])
        for j in x[i]:
            output.write(str(j) + ' ')
        output.write('\n')
    #x = np.array(x)
    output.close()
    print("x done")

def get_vocab_trans(vocabpath='vocab.txt', transpath='trans.txt'):
    vocab = []
    x_text = []
    with open(vocabpath, 'r', encoding='utf-8') as f_vocab:
        vocab = f_vocab.readlines()
        '''for line in vocabs:
            #print(line)
            vocab.append(line)'''
        print(len(vocab))

    with open(transpath, 'r', encoding='utf-8') as f_trans:
        vocabs = f_trans.readlines()
        for line in vocabs:
            l = line.split()
            #print(len(l))
            y = [float(i) for i in l]
            x_text.append(y)
        x_text = np.array(x_text).astype(np.float32)
        print(x_text.shape)
    return len(vocab), vocab, x_text

def get_text_len(transpath='trans.txt'):
    with open(transpath, 'r', encoding='utf-8') as f_trans:
        vocabs = f_trans.readlines()
        text_len =[]
        for line in vocabs:
            l = line.split()
            #print(len(l))
            y = [float(i) for i in l]
            max_len = 0
            for i,k in enumerate(y):
                if k != 0: 
                    max_len = i
            text_len.append(max_len)

        #print(x_text.shape)
        text_len = np.array(text_len)
        print(np.mean(text_len))
    return text_len

if __name__ == "__main__":
    #vocab_encoder_gen("../data/train.csv", "../model/w2v_test_32.model")
    get_vocab_trans()
    get_text_len()