import numpy as np
from theano import function, config, shared, sandbox
import time
import sys
import subprocess
import theano
from theano import tensor as T
import random
import os

import load #to load the training and test data
from e2hattn1 import model #to import the model
from tools import *

import cPickle #ADDED for saving the model using cPickle
import re

def load_params(file_path):

    obj = None
    
    f = file(file_path, 'r')
    obj = cPickle.load(f)
    f.close()
    
    return obj

def save_params(obj, file_path):
    # sanity check
    if file_path == '':
        print 'Give a valid path'
        return
    # sanity check
    path = os.path.dirname(file_path)
    if not os.path.exists(path):
        os.makedirs(path)

    f = file(file_path, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == '__main__':

    s = {'lr' : 0.0627,        #0.0627
         'win': 1,             #Windows for each input word from source sentence
         'nhidden': 1000,      #No of hidden units in hidden layer
         'seed': 345,          #Seed to randomize the training data
         'emb_dimension':500,  ## BE CAREFUL HERE !! 620
         'nepochs':40,         #Max no of training iterations
	 'l':500,              #Maxout hidden layer size in deep output
         'bs':3}

    prevIter = raw_input("Want to continue if there exists any previous iterations? (y/n): ")
    while not (prevIter == 'y' or prevIter == 'Y' or prevIter == 'n' or prevIter == 'N'):
        prevIter = raw_input("Want to continue if there exists any previous iterations? (y/n): ")
    if prevIter == 'y' or prevIter == 'Y':
        prevIter = 1
    else:
        prevIter = 0

    print 'Started Loading Data...'
    #train_set and test_set are training and test parallel sentences respectively
    #target_word2idx and source_word2idx are mappings from word -> index in taget(hindi) and source(english) languages respectively
    #target_idx2vector and source_idx2vector are mappings from index -> vector (word2vec) in taget(hindi) and source(english) languages respectively

    train_set, test_set, target_word2idx, source_idx2vector, source_word2idx, target_idx2vector = load.readFromPklFile('/home/ssarkar/pranay/En_Hin_Corp_4_12_1.pkl.gz') ## BE CAREFUL HERE !!
    print('Loaded data!')
    
    # _lex is source sentence (English)
    # _y is target sentence (Hindi)
    train_lex, train_y = train_set 
    test_lex, test_y = test_set

    print 'Target Vocabulary Size '+str(len(target_word2idx))
    vocsize = len(target_word2idx)  #len(set(reduce(lambda x,y: list(x)+list(y), train_lex+test_lex)))
    nclasses = vocsize              #No of classes to be predicted = Target vocabulary size
    nsentences=len(train_lex)       #nsentences is the No of training sentences
    
    np.random.seed(s['seed'])        
    random.seed(s['seed'])

    print 'Generating embeddings...'
    source_embeddings = [] #c
    for index in source_idx2vector: #c
        source_embeddings.append(source_idx2vector[index]) #c

    target_embeddings = [] #c
    for index in target_idx2vector: #c
        target_embeddings.append(target_idx2vector[index]) #c

    listofzeros = [0.0] * s['emb_dimension']
    target_embeddings.append(listofzeros)
    #print target_embeddings[-1]
    
    source_embedding = theano.shared(np.array(source_embeddings).astype(theano.config.floatX)) #c
    target_embedding = theano.shared(np.array(target_embeddings).astype(theano.config.floatX)) #c
    print 'Generated embeddings!'
    
    rnn = model(n = s['nhidden'],
		        m = s['emb_dimension'],
		        l = s['l'],
 	            	n_y = vocsize,
        	    	source_embeddings=source_embedding,
 		    	target_embeddings=target_embedding)
    print 'Model Object rnn Created'
    
    best_f1= -np.inf
    s['clr'] = s['lr']
    #rnn.normalize_source()
    #rnn.normalize_target()

    for i in xrange(s['nepochs']):
        if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__)))+'/saved_model/model'+str(i)+'.save'):
            break
    currModel = i-1
    
    if prevIter == 1 and currModel >= 0:
        rnn = load_params('/home/ssarkar/pranay/bahdanau_ICLR2015_with_saving/saved_model/model'+str(currModel)+'.save') #loading from the saved file

    for e in xrange(currModel+1,s['nepochs']):
        shuffle([train_lex, train_y], s['seed']) #Random shuffling of training sentences after each epoch
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            source_cwords = contextwin(train_lex[i], s['win']) #c
            train_y_past = np.delete(train_y[i],-1)
            train_y_past = np.insert(train_y_past, 0, -1)
            target_cwords = contextwin(train_y_past, s['win']) #c
            labels = train_y[i]
            rnn.sentence_train(source_cwords, target_cwords, np.asarray(labels).astype('int32'), s['clr']) #c
            #rnn.normalize()
            frac = (i+1)*100./nsentences
            if frac - int(frac) == 0 :
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()
        print 'Training ended for Epoch '+str(e)

        #Saving Model after each epoch
        print 'Saving Model after '+str(e)+'th epoch'
        sys.setrecursionlimit(1000000)
        save_params(rnn, '/home/ssarkar/pranay/bahdanau_ICLR2015_with_saving/saved_model/model'+str(e)+'.save')  #saving 
        #rnn = load_params('/home/ssarkar/pranay/bahdanau_ICLR2015_with_saving/saved_model/model'+str(e)+'.save') #loading from the saved file
        print 'Model Saved!'


        print 'Prediction for Epoch '+ str(e) +' ...'
        prediction_test = []
        alphas_list = []
        print 'started to classify !!'
        for indx in xrange(len(test_lex)): #c
            classified = rnn.classify(np.asarray(contextwin(test_lex[indx], s['win'])).astype('int32')) #c
            pred = classified[0]
            alphas = classified[1]
            prediction_test.append(pred)
            alphas_list.append(alphas)
        print 'Ended classification !!'

        print 'Started writing Translations to Disk...'

        # fout = open('/home/ssarkar/pranay/bahdanau_ICLR2015_with_saving/results/attention_models/bahdanau_2015/'+str(e), 'w') #c
        # print 'opened file'
        # for i in range(len(prediction_test)):
        #     for j in range(len(list(test_lex[i]))):
        #         fout.write(str(list(test_lex[i])[j])+' ')
        #     fout.write('\n')
        #     for j in range(len(test_y[i])):
        #         fout.write(str(test_y[i][j])+' ')
        #     fout.write('\n')
        #     for j in range(len(prediction_test[i])):
        #         fout.write(str(prediction_test[i][j])+' ')
        #     fout.write('\n')

        #     fout.write('__________________________________________________________________________________________________________________________________________\n')
        # fout.close()

        fout = open('/home/ssarkar/pranay/bahdanau_ICLR2015_with_saving/results/attention_models/bahdanau_2015/'+str(e), 'w') #c
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

        falphas = open('/home/ssarkar/pranay/bahdanau_ICLR2015_with_saving/results/attention_models/bahdanau_2015/alphas_clip5_'+str(e), 'w') #c
        print 'opened file'
        for i in range(len(prediction_test)):
            for j in range(len(list(test_lex[i]))):
                falphas.write(str(list(test_lex[i])[j])+' ')
            falphas.write('\n')
            for j in range(len(test_y[i])):
                falphas.write(str(test_y[i][j])+' ')
            falphas.write('\n')
            for j in range(len(prediction_test[i])):
                falphas.write(str(prediction_test[i][j])+' ')
            falphas.write('\n\n')

            alphas = alphas_list[i]
            for alpha in alphas:
                for alpha_x in alpha:
                    falphas.write(str(alpha_x)+' ')
                falphas.write('\n')
            falphas.write('\n')

            falphas.write('__________________________________________________________________________________________________________________________________________\n')
        falphas.close()
    
    
