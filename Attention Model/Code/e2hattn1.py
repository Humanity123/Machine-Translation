import theano
import numpy as np, os
from theano import tensor as T
from collections import OrderedDict
from tools import sample_weights

sigma = lambda x: 1 / (1 + T.exp(-x))

class model(object):

    def __init__(self, n, m, l, n_y, source_embeddings, target_embeddings):

        '''
        n :: dimension of the hidden layer
        n_y :: number of word embeddings in the vocabulary
        m :: dimension of the word embeddings
        n' :: dimension of the hidden layer in alignemnt layer . n'=n
        l  :: maxout hidden layer in the deep output
        '''

        # Define the word embeddings
        self.source_emb = source_embeddings
        self.target_emb = target_embeddings

        #Encoder matrices

        # Forward matrices
        self.Wzf = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Uzf = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Wrf = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Urf = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Wf = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Uf = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.hf0 = theano.shared(np.zeros(n, dtype=theano.config.floatX))
        # Backward matrices
        self.Wzb = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Uzb = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Wrb = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Urb = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Wb = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Ub = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.hb0 = theano.shared(np.zeros(n, dtype=theano.config.floatX))

        # Decoder matrices
        self.Wa = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Ua = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (2*n, n)).astype(theano.config.floatX))

        self.Wzd = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Uzd = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Wrd = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Urd = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Wd = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, n)).astype(theano.config.floatX))
        self.Ud = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))
        self.Cr = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (2*n, n)).astype(theano.config.floatX))
        self.Cz = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (2*n, n)).astype(theano.config.floatX))
        self.C = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (2*n, n)).astype(theano.config.floatX))


        self.W0 = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (l, n_y)).astype(theano.config.floatX))
        self.U0 = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, 2*l)).astype(theano.config.floatX))
        self.C0 = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (2*n, 2*l)).astype(theano.config.floatX))
        self.V0 = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (m, 2*l)).astype(theano.config.floatX))

        self.va = theano.shared(np.cast[theano.config.floatX](np.random.uniform(-0.1,0.1,size = n)))
        self.hd0 = theano.shared(np.zeros(n, dtype=theano.config.floatX))
        self.xt0 = theano.shared(np.zeros(m, dtype=theano.config.floatX))

        self.Ws = theano.shared(0.2* np.random.uniform(-1.0, 1.0, (n, n)).astype(theano.config.floatX))

        source_idxs = T.imatrix()
        target_idxs = T.imatrix()
        xsource = self.source_emb[source_idxs].reshape((source_idxs.shape[0], m))
        xtarget = self.target_emb[target_idxs].reshape((target_idxs.shape[0], m))
        y = T.ivector('y')

        self.params = [self.Wzf, self.Uzf, self.Wrf, self.Urf, self.Wf, self.Uf, self.Wzb, self.Uzb, self.Wrb, self.Urb, self.Wb, self.Ub,self.Wa, self.Ua, self.Wzd, self.Uzd, self.Wrd, self.Urd, self.Wd, self.Ud, self.Cr, self.Cz, self.C0, self.C, self.W0, self.U0, self.V0, self.va]
        self.names = ['Wzf', 'Uzf', 'Wrf', 'Urf', 'Wf', 'Uf', 'Wzb', 'Uzb', 'Wrb', 'Urb', 'Wb', 'Ub', 'Wa', 'Ua', 'Wzd', 'Uzd', 'Wrd', 'Urd', 'Wd', 'Ud', 'Cr', 'Cz', 'C0', 'C', 'W0', 'U0', 'V0', 'va']


        def encoderf(x_t, he_tm1):
            z = sigma(theano.dot(x_t, self.Wzf)+theano.dot(he_tm1, self.Uzf))
            r = sigma(theano.dot(x_t, self.Wrf)+theano.dot(he_tm1, self.Urf))
            ht = T.tanh(theano.dot(x_t, self.Wf) + theano.dot((r*he_tm1), self.Uf))
            h = (1.0 - z)*he_tm1 + z*ht
            return h

        def encoderb(x_t, he_tm1):
            z = sigma(theano.dot(x_t, self.Wzb)+theano.dot(he_tm1, self.Uzb))
            r = sigma(theano.dot(x_t, self.Wrb)+theano.dot(he_tm1, self.Urb))
            ht = T.tanh(theano.dot(x_t, self.Wb) + theano.dot((r*he_tm1), self.Ub))
            h = (1.0 - z)*he_tm1 + z*ht
            return h

        hf,_ = theano.scan(fn=encoderf, sequences=xsource, outputs_info=self.hf0, non_sequences=None, n_steps=xsource.shape[0])
        hb,_ = theano.scan(fn=encoderb, sequences=xsource[::-1], outputs_info=self.hb0, non_sequences=None, n_steps=xsource.shape[0])

        h = T.concatenate([hf, hb[::-1]], axis=1)

        def decoder(w_t, st_m1, h):
            e = T.dot(self.va,(T.tanh(theano.dot(st_m1,self.Wa)+theano.dot(h,self.Ua))).dimshuffle(1,0))
            alpha = (T.exp(e))/((T.exp(e)).sum())
            c = theano.dot(alpha, h)
            z = sigma(theano.dot(w_t,self.Wzd) + theano.dot(st_m1, self.Uzd) + theano.dot(c, self.Cz))
            r = sigma(theano.dot(w_t,self.Wrd) + theano.dot(st_m1, self.Urd) + theano.dot(c, self.Cr))
            st = T.tanh(theano.dot(w_t,self.Wd) + theano.dot(r*st_m1, self.Ud) + theano.dot(c, self.C))
            s = (1 - z)*st_m1 + z*st
            tt = sigma(theano.dot(s,self.U0) + theano.dot(w_t, self.V0) + theano.dot(c, self.C0)) # length 2l
            # Maxout layer for : t -> length l
            ttr = tt.reshape((l,2))
            t = None
            for i in xrange(2):
                tmp = ttr[:,i::2]
                if t is None:
                    t = tmp
                else:
                    t = T.maximum(t, tmp)
            y_t = (T.exp(theano.dot(t.dimshuffle(1,0), self.W0))/(T.exp(theano.dot(t.dimshuffle(1,0), self.W0)).sum())).dimshuffle(1,0).flatten()
            return [s, y_t]

        [sd, yt], _ = theano.scan(fn=decoder, sequences=xtarget, outputs_info=[self.hd0, None], non_sequences=h, n_steps=y.shape[0])

        p_y_given_sentence = yt
        lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_sentence)[T.arange(y.shape[0]), y])
        sentence_gradients=T.grad(sentence_nll, self.params)
        # Implement gradient clipping  - start
        grads = sentence_gradients
        clip_c = 5.0
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(T.switch(g2 > (clip_c**2), g / T.sqrt(g2) * clip_c, g))
            grads = new_grads
        sentence_gradients = grads
        # Implement gradient clipping  - end
        sentence_updates=OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , sentence_gradients))
        self.sentence_train=theano.function(inputs=[source_idxs, target_idxs, y, lr], outputs=sentence_nll, updates=sentence_updates)

        # Decoder test start

        def decoder_test(st_m1, w_t, h):
            e = T.dot(self.va,(T.tanh(theano.dot(st_m1,self.Wa)+theano.dot(h,self.Ua))).dimshuffle(1,0))
            alpha = (T.exp(e))/((T.exp(e)).sum())
            c = theano.dot(alpha, h)
            z = sigma(theano.dot(w_t,self.Wzd) + theano.dot(st_m1, self.Uzd) + theano.dot(c, self.Cz))
            r = sigma(theano.dot(w_t,self.Wrd) + theano.dot(st_m1, self.Urd) + theano.dot(c, self.Cr))
            st = T.tanh(theano.dot(w_t,self.Wd) + theano.dot(r*st_m1, self.Ud) + theano.dot(c, self.C))
            s = (1 - z)*st_m1 + z*st
            tt = sigma(theano.dot(s,self.U0) + theano.dot(w_t, self.V0) + theano.dot(c, self.C0)) # length 2l
            # Maxout layer for : t -> length l
            ttr = tt.reshape((l,2))
            t = None
            for i in xrange(2):
                tmp = ttr[:,i::2]
                if t is None:
                    t = tmp
                else:
                    t = T.maximum(t, tmp)
            y_t = (T.exp(theano.dot(t.dimshuffle(1,0), self.W0))/(T.exp(theano.dot(t.dimshuffle(1,0), self.W0)).sum())).dimshuffle(1,0).flatten()
            y_test_pred = T.argmax(y_t)
            x_prev = self.target_emb[y_test_pred]
            return [s, y_t, x_prev, alpha], theano.scan_module.until(T.eq(y_test_pred, 1)) # Change the index to EOS for the language. (61 for be_hi and 1 for en to hi according to embeddings)

        [stest, ytest, xprev, alphas], _ = theano.scan(fn=decoder_test, sequences=None, outputs_info=[self.hd0, None, self.xt0, None], non_sequences=h, n_steps=(xsource.shape[0]*2))
        p_y_given_test_sentence = ytest
        y_test_pred = T.argmax(p_y_given_test_sentence, axis=1)
        self.classify = theano.function(inputs=[source_idxs], outputs=[y_test_pred, alphas])
        # Decoder test end
        self.normalize_source = theano.function( inputs = [],
                         updates = {self.source_emb:\
                         self.source_emb/T.sqrt((self.source_emb**2).sum(axis=1)).dimshuffle(0,'x')})
        self.normalize_target = theano.function( inputs = [],
                         updates = {self.target_emb:\
                         self.target_emb/T.sqrt((self.target_emb**2).sum(axis=1)).dimshuffle(0,'x')})

        def save(self, folder):
            for param, name in zip(self.params, self.names):
                np.save(os.path.join(folder, name + '.npy'), param.get_value())

