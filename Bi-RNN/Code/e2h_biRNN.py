
import theano
from theano import function, config, shared, sandbox
import numpy, os
from theano import tensor as T
from theano.compat import OrderedDict
import load


class model(object):

    def __init__(self, nh,nc,ne,de,cs,embedding):

        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''

        # Define the weight embeddings and matrices
        self.emb = embedding
        #self.emb = theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))
        
        #Encoder matrices
        self.Wx=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX)) # Between input and enc forward hidden layer
        self.Wxb=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX)) #Between input and enc backward hidden layer
        self.Whe=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)) # Encoder forward hidden layer feedback
        self.Wheb=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)) # Encoder backward hidden layer feedback
        
        #Decoder matrices
        self.Wy=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (ne, nh)).astype(theano.config.floatX)) # Between previous o/p and decode hl
        self.Wc=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)) # Between c and decoder hl
        self.Wcb=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)) # Between cb and decoder hl
        self.Whd=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)) # Decoder hidden layer feedback
        self.W=theano.shared(0.2* numpy.random.uniform(-1.0, 1.0, (nh, ne)).astype(theano.config.floatX)) # Between decoder hl and final o/p layer


        self.he0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.heb0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.hd0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        self.s0 = theano.shared(numpy.zeros(ne, dtype=theano.config.floatX))
        #self.s0=theano.shared(numpy.zeros((nc, nc)).astype(theano.config.floatX))

        self.params = [self.Wx, self.Wxb, self.Whe, self.Wheb, self.Wy, self.Wc, self.Wcb, self.Whd, self.W, self.he0, self.heb0, self.hd0, self.s0]
        self.names = ['Wx', 'Wxb', 'Whe', 'Wheb', 'Wy', 'Wc', 'Wcb', 'Whd', 'W', 'he0', 'heb0', 'hd0', 's0']

        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        xb = x[::-1]
        y = T.ivector('y')

        def encoder(x_t, he_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(he_tm1, self.Whe))
            print 'shape ht', h_t.ndim
            return h_t

        def encoder_back(xb_t, heb_tm1):
            hb_t = T.nnet.sigmoid(T.dot(xb_t, self.Wxb) + T.dot(heb_tm1, self.Wheb))
            print 'shape hbt', hb_t.ndim
            return hb_t

        c,_ = theano.scan(fn=encoder, sequences=x, outputs_info=self.he0, non_sequences=None, n_steps=x.shape[0])
        cb,_ = theano.scan(fn=encoder_back, sequences=xb, outputs_info=self.heb0, non_sequences=None, n_steps=x.shape[0])

        def decoder(hd_tm1, s_tm1, c1, cb1):
            hd_t = T.nnet.sigmoid(T.dot(hd_tm1, self.Whd) + T.dot(s_tm1, self.Wy) + T.dot(c1, self.Wc) + T.dot(cb1, self.Wcb))
            s_t = T.exp(T.dot(hd_t, self.W))/(T.exp(T.dot(hd_t, self.W)).sum())
            return [hd_t, s_t]

        [h, s], _ = theano.scan(fn=decoder, outputs_info=[self.hd0, self.s0], non_sequences=[c[-1], cb[-1]],n_steps=x.shape[0])
        print 's = ', s.ndim
        p_y_given_sentence = s
        lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_sentence)[T.arange(y.shape[0]), y])
        sentence_gradients=T.grad(sentence_nll, self.params)
        sentence_updates=OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , sentence_gradients))
        self.sentence_train=theano.function(inputs=[idxs, y, lr], outputs=sentence_nll, updates=sentence_updates)

        def decoder_test(hd_tm1, s_tm1, c1, cb1):
            hd_t = T.nnet.sigmoid(T.dot(hd_tm1, self.Whd) + T.dot(s_tm1, self.Wy) + T.dot(c1, self.Wc) + T.dot(cb1, self.Wcb))
            s_t = T.exp(T.dot(hd_t, self.W))/(T.exp(T.dot(hd_t, self.W)).sum())
            y_test_pred = T.argmax(s_t)
            return [hd_t, s_t], theano.scan_module.until(T.eq(y_test_pred ,1))

        [htest, stest], _ = theano.scan(fn=decoder_test, outputs_info=[self.hd0, self.s0], non_sequences=[c[-1], cb[-1]], n_steps=(x.shape[0]*4))
        p_y_given_test_sentence = stest
        y_test_pred = T.argmax(p_y_given_test_sentence, axis=1)
        

        self.classify = theano.function(inputs=[idxs], outputs=y_test_pred)
        
        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
