
import theano
import numpy as np, os
from theano import tensor as T
from collections import OrderedDict
from tools import sample_weights

sigma = lambda x: 1 / (1 + T.exp(-x))

class model(object):

    def __init__(self, n_hidden,n_c,n_i, n_o, n_f, n_y, de,cs,source_embedding,target_embedding):

        '''
        n_hidden :: dimension of the hidden layer
        n_c :: dimension of the memory
        n_y :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''

        # Define the weight embeddings and matrices
	self.source_emb = source_embedding #c
        self.target_emb = target_embedding #c
        n_in = de*cs
	lambda_2 = 0.001
        #Encoder matrices
        self.We_xi = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n_in, n_i)).astype(theano.config.floatX))
        self.We_hi = theano.shared(sample_weights(n_hidden, n_i))
        self.be_i = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.5,.5,size = n_i)))
        self.We_xf = theano.shared(sample_weights(n_in, n_f))
        self.We_hf = theano.shared(sample_weights(n_hidden, n_f))
        self.be_f = theano.shared(np.cast[theano.config.floatX](np.random.uniform(0, 1.,size = n_f)))
        self.We_xc = theano.shared(sample_weights(n_in, n_c))
        self.We_hc = theano.shared(sample_weights(n_hidden, n_c))
        self.be_c = theano.shared(np.zeros(n_c, dtype=theano.config.floatX))
        self.We_xo = theano.shared(sample_weights(n_in, n_o))
        self.We_ho = theano.shared(sample_weights(n_hidden, n_o))
        self.be_o = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.5,.5,size = n_o)))

        self.he0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        self.ce0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        source_idxs = T.imatrix()
	target_idxs = T.imatrix()
        xsource = self.source_emb[source_idxs].reshape((source_idxs.shape[0], n_in))
        xtarget = self.target_emb[target_idxs].reshape((target_idxs.shape[0], n_in))
        y = T.ivector('y')


        #Decoder matrices
        self.Wd_yi = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n_in, n_i)).astype(theano.config.floatX))
        self.Wd_hi = theano.shared(sample_weights(n_hidden, n_i))
        self.Wd_heTi = theano.shared(sample_weights(n_hidden, n_i))
        self.bd_i = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.5,.5,size = n_i)))
        self.Wd_yf = theano.shared(sample_weights(n_in, n_f))
        self.Wd_hf = theano.shared(sample_weights(n_hidden, n_f))
        self.Wd_heTf = theano.shared(sample_weights(n_hidden, n_f))
        self.bd_f = theano.shared(np.cast[theano.config.floatX](np.random.uniform(0, 1.,size = n_f)))
        self.Wd_yc = theano.shared(sample_weights(n_in, n_c))
        self.Wd_hc = theano.shared(sample_weights(n_hidden, n_c))
        self.Wd_heTc = theano.shared(sample_weights(n_hidden, n_c))
        self.bd_c = theano.shared(np.zeros(n_c, dtype=theano.config.floatX))
        self.Wd_yo = theano.shared(sample_weights(n_in, n_o))
        self.Wd_ho = theano.shared(sample_weights(n_hidden, n_o))
        self.Wd_heTo = theano.shared(sample_weights(n_hidden, n_o))
        self.bd_o = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.5,.5,size = n_o)))
        self.W = theano.shared(sample_weights(n_hidden, n_y))
        self.b = theano.shared(np.zeros(n_y, dtype=theano.config.floatX))

        self.hd0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        self.cd0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        self.s0 = theano.shared(np.zeros(n_y, dtype=theano.config.floatX))
	self.xt0 = theano.shared(np.zeros(n_in, dtype=theano.config.floatX))

        self.params = [self.We_xi, self.We_hi, self.be_i, self.We_xf, self.We_hf, self.be_f, self.We_xo, self.We_ho, self.be_o, self.We_xc, self.We_hc, self.be_c, self.he0, self.ce0, self.Wd_yi, self.Wd_hi, self.Wd_heTi, self.bd_i, self.Wd_yf, self.Wd_hf, self.Wd_heTf, self.bd_f, self.Wd_yo, self.Wd_ho, self.Wd_heTo, self.bd_o, self.Wd_yc, self.Wd_hc, self.Wd_heTc, self.bd_c, self.W, self.b, self.hd0, self.cd0]
        self.names = ['We_xi', 'We_hi', 'be_i', 'We_xf', 'We_hf', 'be_f', 'We_xo', 'We_ho','be_o', 'We_xc', 'We_hc', 'be_c', \
                       'he0', 'ce0', 'Wd_yi', 'Wd_hi', 'Wd_heTi', 'bd_i', 'Wd_yf', '.Wd_hf', 'Wd_heTf', 'bd_f',\
                       'Wd_yo', 'Wd_ho', 'Wd_heTo', 'bd_o', 'Wd_yc', 'Wd_hc', 'Wd_heTc', 'bd_c', 'W', 'b',\
                       'hd0', 'cd0']
	#self.L2 = (self.We_xi ** 2).sum() + (self.We_hi ** 2).sum() + (self.be_i ** 2).sum() + (self.We_xf ** 2).sum() + (self.We_hf ** 2).sum() + (self.be_f ** 2).sum() + (self.We_xo ** 2).sum() + (self.We_ho ** 2).sum() + (self.be_o ** 2).sum() + (self.We_xc ** 2).sum() + (self.We_hc ** 2).sum() + (self.be_c ** 2).sum() + (self.he0 ** 2).sum() + (self.ce0 ** 2).sum() + (self.Wd_yi  ** 2).sum() +(self.Wd_hi ** 2).sum() + (self.Wd_heTi ** 2).sum() + (self.bd_i ** 2).sum() + (self.Wd_yf ** 2).sum() + (self.Wd_hf ** 2).sum() + (self.Wd_heTf ** 2).sum() + (self.bd_f ** 2).sum() + (self.Wd_yo ** 2).sum() + (self.Wd_ho ** 2).sum() + (self.Wd_heTo ** 2).sum() + (self.bd_o ** 2).sum() + (self.Wd_yc ** 2).sum() + (self.Wd_hc ** 2).sum() + (self.Wd_heTc ** 2).sum() + (self.bd_c ** 2).sum() + (self.W ** 2).sum() + (self.b ** 2).sum() + (self.hd0 ** 2).sum() + (self.cd0 ** 2).sum()
	
        def encoder(x_t, he_tm1, ce_tm1):
            i_t = sigma(theano.dot(x_t, self.We_xi) + theano.dot(he_tm1, self.We_hi) + self.be_i)
            f_t = sigma(theano.dot(x_t, self.We_xf) + theano.dot(he_tm1, self.We_hf) + self.be_f)
            o_t = sigma(theano.dot(x_t, self.We_xo)+ theano.dot(he_tm1, self.We_ho) + self.be_o)
            c_t = f_t * ce_tm1 + i_t * T.tanh(theano.dot(x_t, self.We_xc) + theano.dot(he_tm1, self.We_hc) + self.be_c)
            he_t = o_t * T.tanh(c_t)
            return [he_t, c_t]

        [he, c],_ = theano.scan(fn=encoder, sequences=xsource, outputs_info=[self.he0, self.ce0], non_sequences=None,n_steps=xsource.shape[0])

        def decoder(x, hd_tm1, cd_tm1, c1):
            i_t = sigma(theano.dot(x, self.Wd_yi) + theano.dot(hd_tm1, self.Wd_hi) + theano.dot(c1, self.Wd_heTi) + self.bd_i)
            f_t = sigma(theano.dot(x, self.Wd_yf) + theano.dot(hd_tm1, self.Wd_hf) + theano.dot(c1, self.Wd_heTf) + self.bd_f)
            o_t = sigma(theano.dot(x, self.Wd_yo)+ theano.dot(hd_tm1, self.Wd_ho)  + theano.dot(c1, self.Wd_heTo) + self.bd_o)
            c_t = f_t * cd_tm1 + i_t * T.tanh(theano.dot(x, self.Wd_yc) + theano.dot(hd_tm1, self.Wd_hc) + theano.dot(c1, self.Wd_heTc) + self.bd_c)
            hd_t = o_t * T.tanh(c_t)
            y_t = T.exp(T.dot(hd_t, self.W) + self.b)/(T.exp(T.dot(hd_t, self.W) + self.b).sum())
            return [hd_t, y_t, c_t]

        [h, s, cd], _ = theano.scan(fn=decoder, sequences=xtarget, outputs_info=[self.hd0, None, self.cd0], non_sequences=he[-1],n_steps=y.shape[0])

        p_y_given_sentence = s
        #y_pred = T.argmax(p_y_given_sentence, axis=1)
        lr = T.scalar('lr')
	#L2 = T.sum(self.params ** 2)
        sentence_nll = -T.mean(T.log(p_y_given_sentence)[T.arange(y.shape[0]), y]) # + lambda_2*self.L2

        sentence_gradients=T.grad(sentence_nll, self.params)
        sentence_updates=OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , sentence_gradients))
        self.sentence_train=theano.function(inputs=[source_idxs, target_idxs, y, lr], outputs=sentence_nll, updates=sentence_updates)

        # Decoder test start

        def decoder_test(hd_tm1, cd_tm1, x, c1):
            i_t = sigma(theano.dot(x, self.Wd_yi) + theano.dot(hd_tm1, self.Wd_hi) + theano.dot(c1, self.Wd_heTi) + self.bd_i)
            f_t = sigma(theano.dot(x, self.Wd_yf) + theano.dot(hd_tm1, self.Wd_hf) + theano.dot(c1, self.Wd_heTf) + self.bd_f)
            o_t = sigma(theano.dot(x, self.Wd_yo)+ theano.dot(hd_tm1, self.Wd_ho)  + theano.dot(c1, self.Wd_heTo) + self.bd_o)
            c_t = f_t * cd_tm1 + i_t * T.tanh(theano.dot(x, self.Wd_yc) + theano.dot(hd_tm1, self.Wd_hc) + theano.dot(c1, self.Wd_heTc) + self.bd_c)
            hd_t = o_t * T.tanh(c_t)
            y_t = T.exp(T.dot(hd_t, self.W) + self.b)/(T.exp(T.dot(hd_t, self.W) + self.b).sum())
            y_test_pred = T.argmax(y_t)
	    x_prev = self.target_emb[y_test_pred]
            return [hd_t, y_t, c_t, x_prev], theano.scan_module.until(T.eq(y_test_pred, 0))

        [htest, stest, cdtest, xprev], _ = theano.scan(fn=decoder_test, sequences=None, outputs_info=[self.hd0, None, self.cd0, self.xt0], non_sequences=he[-1], n_steps=(xsource.shape[0]*4))
        p_y_given_test_sentence = stest
        y_test_pred = T.argmax(p_y_given_test_sentence, axis=1)
        self.classify = theano.function(inputs=[source_idxs], outputs=y_test_pred)
        # Decoder test end
        self.normalize = theano.function( inputs = [],
                         updates = {self.source_emb:\
                         self.source_emb/T.sqrt((self.source_emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
