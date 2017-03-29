import numpy as np
from theano import function, config, shared, sandbox
import time
import sys
import subprocess
import theano
from theano import tensor as T
import random

import load
from e2e_lstm import model
from tools import *

if __name__ == '__main__':

    s = {'lr' : 0.0627,
         'win': 1,
         'nhidden': 1000,
         'seed': 345,
         'emb_dimension':620,
         'nepochs':40,
         'bs':3}

    print 'hello'
    train_set, test_set, target_word2idx, source_idx2vector, source_word2idx, target_idx2vector = load.readFromPklFile('/home/ssarkar/work/En_Hin_Corp.pkl.gz')
    print('Read data !!')
    train_lex, train_y = train_set
    test_lex, test_y = test_set
    print('find vocsize')
    print len(target_word2idx)
    vocsize = len(target_word2idx)#len(set(reduce(lambda x,y: list(x)+list(y), train_lex+test_lex)))
    nclasses = vocsize
    nsentences=len(train_lex)
    np.random.seed(s['seed'])
    random.seed(s['seed'])

    source_embeddings = [] #c
    for index in source_idx2vector: #c
        source_embeddings.append(source_idx2vector[index]) #c

    target_embeddings = [] #c
    for index in target_idx2vector: #c
        target_embeddings.append(target_idx2vector[index]) #c
    listofzeros = [0.0] * s['emb_dimension']
    target_embeddings.append(listofzeros)
    print target_embeddings[-1]
    

    source_embedding = theano.shared(np.array(source_embeddings).astype(theano.config.floatX)) #c
    target_embedding = theano.shared(np.array(target_embeddings).astype(theano.config.floatX)) #c

    print 'Generated embeddings !!'

    rnn = model( n_hidden = s['nhidden'],
                 n_c = s['nhidden'],
                 n_i = s['nhidden'],
                 n_o = s['nhidden'],
                 n_f = s['nhidden'],
                 n_y = vocsize,
                 de = s['emb_dimension'],
                 cs = s['win'],
                 source_embedding=source_embedding,
		 target_embedding=target_embedding)

    best_f1= -np.inf
    s['clr'] = s['lr']
    #rnn.normalize()
    for e in xrange(s['nepochs']):
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
             source_cwords = contextwin(train_lex[i], s['win']) #c
             source_words=map(lambda x: np.asarray(x).astype('int32'), minibatch(source_cwords,len(source_cwords))) #c
	     train_y_past = np.delete(train_y[i],-1)
	     train_y_past = np.insert(train_y_past, 0, -1)
	     #target_cwords = contextwin(train_y[i], s['win']) #c
	     target_cwords = contextwin(train_y_past, s['win']) #c
             target_words=map(lambda x: np.asarray(x).astype('int32'), minibatch(target_cwords,len(target_cwords))) #c

             labels = train_y[i]
             #for word_batch in words:
                 #label_set = labels[0:len(word_batch)]
                 #rnn.sentence_train(word_batch, np.asarray(label_set).astype('int32'), s['clr'])
                 #rnn.normalize()
             rnn.sentence_train(source_cwords, target_cwords, np.asarray(labels).astype('int32'), s['clr']) #c
             rnn.normalize()
             #print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
             sys.stdout.flush()
        prediction_test = []
        for sentence in test_lex:
            #print 'start classify !!'
            pred = rnn.classify(np.asarray(contextwin(sentence, s['win'])).astype('int32'))
            #print 'end classify !!'
            prediction_test.append(pred)
        print 'start writing'
        fout = open('/home/ssarkar/work/results/'+str(e), 'w')
        print 'opened file'
        for i in range(len(prediction_test)):
            for j in range(len(list(test_lex[i]))):
                fout.write(str(list(test_lex[i])[j])+' ')
            fout.write('\n')
            for j in range(len(test_y[i])):
                fout.write(str(test_y[i][j])+' ')
            fout.write('\n')
            for j in range(len(prediction_test[i])):
                fout.write(str(prediction_test[i][j])+' ')
            fout.write('\n')

            fout.write('__________________________________________________________________________________________________________________________________________\n')
        fout.close()
