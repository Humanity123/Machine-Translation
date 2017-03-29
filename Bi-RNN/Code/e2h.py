
import numpy as np
from theano import function, shared, sandbox#, config
import time
import sys
import subprocess
import theano
from theano import tensor as T
import random

import load
from e2h_rnn import model
from tools import *

if __name__ == '__main__':

    s = {'lr' : 0.0627,
         'win': 1,
         'nhidden': 2000,
         'seed': 345,
         'emb_dimension':500,
         'nepochs':1000,
         'bs':3}

    print 'Starting English->Hindi BiRNN'
    train_set, test_set, word2idx, idx2vector = load.readFromPklFile('../En_Hin_Corp_4_12.pkl.gz')
    print('Read data !!')
    train_lex, train_y = train_set
    test_lex, test_y = train_set
    print('find vocsize')
    print len(word2idx)
    vocsize = len(word2idx)#len(set(reduce(lambda x,y: list(x)+list(y), train_lex+test_lex)))
    nclasses = vocsize
    nsentences=len(train_lex)
    np.random.seed(s['seed'])
    random.seed(s['seed'])

    embeddings = []
    for index in idx2vector:
        embeddings.append(idx2vector[index])
    print embeddings[0]

    embedding = theano.shared(np.array(embeddings).astype(theano.config.floatX))
    print 'Generated embeddings !!'
    rnn = model( nh = s['nhidden'],
                 nc = nclasses,
                 ne = vocsize,
                 de = s['emb_dimension'],
                 cs = s['win'],
                 embedding=embedding)

    best_f1= -np.inf
    s['clr'] = s['lr']
    #rnn.normalize()
    for e in xrange(s['nepochs']):
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
             cwords = contextwin(train_lex[i], s['win'])
             words=map(lambda x: np.asarray(x).astype('int32'), minibatch(cwords,len(cwords)))

             labels = train_y[i]
             #for word_batch in words:
                 #label_set = labels[0:len(word_batch)]
                 #rnn.sentence_train(word_batch, np.asarray(label_set).astype('int32'), s['clr'])
                 #rnn.normalize()
             rnn.sentence_train(cwords, np.asarray(labels).astype('int32'), s['clr'])
             rnn.normalize()
             print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
             sys.stdout.flush()
               #exit(0)
        prediction_test = []
        for sentence in test_lex:
            print 'start classify !!'
            pred = rnn.classify(np.asarray(contextwin(sentence, s['win'])).astype('int32'))
            print 'end classify !!'
            prediction_test.append(pred)
        print 'start writing'
        fout = open('BiRNN_results/'+str(e), 'w')
        print 'opened file'
        for i in range(0, len(prediction_test)):
            #print 'writing'
            for j in range(0, len(test_y[i])):
                fout.write(str(test_y[i][j])+' -- '+str(prediction_test[i][j])+',')
            fout.write('\n')
        fout.close()
