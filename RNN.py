"""
Recurrent Neural Network for synthesizing English text character by character
a_t = W*h_(t−1) + U*x_t + b
h_t = tanh(a_t)
o_t = V*h_t + c
p_t = SoftMax(o_t)
"""
# import libraries needed
import numpy as np
import matplotlib.pyplot as plt

# create data class acts as struct to keep data from text file
class Data:
    # pass
    filename, text_data = '', ''
    text_size, chars_size = 0, 0
    char_to_ind, ind_to_char, unique_chars = {}, {}, {}
# setup parameters including coefficient matrices W,U,V,b,c
# create parameter class acts as struct to keep parameters
class Params:
    # pass
    hidden_states, seq_length, eta, sig = 0, 0, 0, 0
    W, U, V, b, c = [], [], [], [], []

# Recurrent Neural Network
class RNN:
    def __init__(self):
        self.data = Data()
        self.params = Params()
    # set parameters
    def setParams(self,hidden_states_,seq_length_,eta_,sig_):
        self.params.hidden_states = hidden_states_
        self.params.seq_length = seq_length_
        self.params.eta = eta_
        self.params.sig = sig_
    # load text data
    def LoadData(self,filename_):
        self.data.filename = filename_
        self.data.text_data = open(self.data.filename, 'r').read()
        self.data.unique_chars = list(set(self.data.text_data))
        self.data.text_size, self.data.chars_size = len(self.data.text_data), len(self.data.unique_chars)
        print('first chars: %s' %(self.data.text_data[0:10]))
        print('text has %d characters, %d unique.' % (self.data.text_size, self.data.chars_size))
        self.data.char_to_ind = {ch:i for i, ch in enumerate(self.data.unique_chars)}
        self.data.ind_to_char = {i:ch for i, ch in enumerate(self.data.unique_chars)}

    def networkInit(self):
        self.params.W = np.random.rand(self.params.hidden_states,self.params.hidden_states)*self.params.sig
        self.params.U = np.random.rand(self.params.hidden_states,self.data.chars_size)*self.params.sig
        self.params.V = np.random.rand(self.data.chars_size,self.params.hidden_states)*self.params.sig
        self.params.b = np.zeros((self.params.hidden_states,1))
        self.params.c = np.zeros((self.data.chars_size,1))

    # synthesize text from randomly initialized self
    def synText(self,h,x0,n):
        """
        h as the hidden state vector at current time
        xo represents the first(dummy) input vector to the self
        n is the length of sequence you want to generate. 
        At every time step,it will generate a vector of probabilities for the labels, and then sample a lable form the
        distribution. 
        :return: The output is a vector of index of all synthesized text
        """
        x = np.zeros((self.data.chars_size,1))
        x[x0] = 1
        index=[]
        for i in range(n):
            h = np.tanh(np.dot(self.params.U,x) + np.dot(self.params.W,h) + self.params.b)
            o = np.dot(self.params.V,h) + self.params.c
            p = np.exp(o)/np.sum(np.exp(o))
            xnext = np.random.choice(range(self.data.chars_size), p=p.ravel())
            x = np.zeros((self.data.chars_size,1)) # next input vector
            x[xnext] = 1
            index.append(xnext)
        return index

    # evaluate the network, forward pass
    def forwardPass(self,textPrev,textNext, hprev):
        """
        
        hprev is initial hidden state
        :return: xs, hs, ps, loss
        """
        xs, hs, os, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        for i in range(len(textPrev)):
            xs[i] = np.zeros((self.data.chars_size,1))
            xs[i][textPrev[i]] = 1
            hs[i] = np.tanh(np.dot(self.params.U,xs[i]) + np.dot(self.params.W,hs[i-1]) + self.params.b)
            os[i] = np.dot(self.params.V,hs[i]) +self.params.c
            ps[i] = np.exp(os[i])/np.sum(np.exp(os[i]))
            loss += -np.log(ps[i][textNext[i],0])
        return xs,hs,ps,loss
    # compute gradients, backward pass
    def backwardPass(self,xs, hs, ps,textPrev,textNext):
        dW, dU, dV = np.zeros_like(self.params.W), np.zeros_like(self.params.U), np.zeros_like(self.params.V)
        db, dc, dhnext = np.zeros_like(self.params.b), np.zeros_like(self.params.c), np.zeros_like(hs[0])
        for i in reversed(range(len(textPrev))):
            do = np.copy(ps[i])
            do[textNext[i]] -= 1
            dh = np.dot(self.params.V.T, do) + dhnext # dhnext=W*da of i+1
            da = (1 - hs[i]**2)*dh
            dhnext = np.dot(self.params.W.T, da)
            dW += np.dot(da,hs[i-1].T)
            dU += np.dot(da,xs[i].T)
            dV += np.dot(do, hs[i].T)
            db += da
            dc += do
            # do = np.copy(ps[i])
            # do[textNext[i]] -= 1
            # dV += np.dot(do, hs[i].T)
            # dc += do
            # dh = np.dot(self.params.V.T, do) + dhnext
            # da = (1 - hs[i]**2)*dh
            # db += da
            # dU += np.dot(da,xs[i].T)
            # dW += np.dot(da,hs[i-1].T)
            # dhnext = np.dot(self.params.W.T, da)
        for tmp in [dW, dU, dV, db, dc]:
            np.clip(tmp,-5,5,out=tmp)#max(min(tmp,5),-5)
        return dW, dU, dV, db, dc, hs[len(textPrev)-1]
    #Adagrad update
    def adagrad(self, dW, dU, dV, db, dc, mW, mU, mV, mb, mc):
        '''
        
        :param dW: 
        :param dU: 
        :param dV: 
        :param db: 
        :param dc: 
        :return: 
        '''
        for param, dparam, mparam in zip([self.params.W, self.params.U, self.params.V, self.params.b, self.params.c],
                                         [dW, dU, dV, db, dc], [mW, mU, mV, mb, mc]):
            mparam += dparam * dparam
            param += -self.params.eta * dparam / np.sqrt(mparam + 1e-8)

    def RMSProp(self, dW, dU, dV, db, dc, mW, mU, mV, mb, mc,gamma):
        '''

        :param dW: 
        :param dU: 
        :param dV: 
        :param db: 
        :param dc: 
        :return: 
        '''
        for param, dparam, mparam in zip([self.params.W, self.params.U, self.params.V, self.params.b, self.params.c],
                                         [dW, dU, dV, db, dc], [mW, mU, mV, mb, mc]):
            mparam = gamma*mparam + (1-gamma)*dparam*dparam
            param += -self.params.eta * dparam / np.sqrt(mparam + 1e-8)

    def adadelta(self, dW, dU, dV, db, dc, dWprev, dUprev, dVprev, dbprev, dcprev,
                 delWprev, delUprev, delVprev, delbprev, delcprev,gamma):
        '''
        
        :param dW: 
        :param dU: 
        :param dV: 
        :param db: 
        :param dc: 
        :param mW: 
        :param mU: 
        :param mV: 
        :param mb: 
        :param mc: 
        :return: 
        '''
        mW, mU, mV, mb, mc = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U),\
                                             np.zeros_like(rnn.params.V),np.zeros_like(rnn.params.b),\
                                             np.zeros_like(rnn.params.c)
        mW2, mU2, mV2, mb2, mc2 = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U),\
                                             np.zeros_like(rnn.params.V),np.zeros_like(rnn.params.b),\
                                             np.zeros_like(rnn.params.c)
        delWnow, delUnow, delVnow, delbnow, delcnow = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U),\
                                             np.zeros_like(rnn.params.V),np.zeros_like(rnn.params.b),\
                                             np.zeros_like(rnn.params.c)
        for param, gnow, gprev, delNow,delPrev, mparam1, mparam2 in zip([self.params.W, self.params.U, self.params.V, self.params.b, self.params.c],
                                         [dW, dU, dV, db, dc], [dWprev, dUprev, dVprev, dbprev, dcprev],
                                    [delWnow, delUnow, delVnow, delbnow, delcnow],
                                    [delWprev, delUprev, delVprev, delbprev, delcprev],
                                                                [mW, mU, mV, mb, mc], [mW2, mU2, mV2, mb2, mc2]):

            mparam1 = gamma*gprev + (1-gamma)*gnow*gnow # current gradient g^2_t=gamma*g^2_(t-1)+(1-gamma)g^2_t
            delNow  = -np.sqrt(mparam2+1e-8)*gnow/np.sqrt(mparam1+ 1e-8)# current gradient's derivative respect to param: delta(param) = -sqrt(delPrev+eps)/sqrt(mparam1+esp)*dnow
            mparam2 = gamma*delPrev + (1-gamma)*delNow*delNow # exponential decaying average of delta of params.
            param += delNow
        return mW, mU, mV, mb, mc, mW2, mU2, mV2, mb2, mc2

if  __name__ == "__main__":
    np.random.seed(0)
    rnn = RNN()
    rnn.setParams(100, 25, 0.07,0.01)
    rnn.LoadData('data/goblet_book.txt')
    rnn.networkInit()
    smooth_loss = -np.log(1/rnn.data.chars_size)*rnn.params.seq_length
    i, e, epoch = 0, 0, 0
    mW, mU, mV, mb, mc = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U), np.zeros_like(rnn.params.V),\
                         np.zeros_like(rnn.params.b), np.zeros_like(rnn.params.c)
    dW, dU, dV, db, dc = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U), np.zeros_like(rnn.params.V),\
                         np.zeros_like(rnn.params.b), np.zeros_like(rnn.params.c)
    dWprev, dUprev, dVprev, dbprev, dcprev = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U),\
                                             np.zeros_like(rnn.params.V),np.zeros_like(rnn.params.b),\
                                             np.zeros_like(rnn.params.c)
    delWprev, delUprev, delVprev, delbprev, delcprev = np.zeros_like(rnn.params.W), np.zeros_like(rnn.params.U),\
                                             np.zeros_like(rnn.params.V),np.zeros_like(rnn.params.b),\
                                             np.zeros_like(rnn.params.c)

    smoothLosses = []
    while True:
        if e + rnn.params.seq_length + 1 >= len(rnn.data.text_data) or i == 0:
            hprev = np.zeros((rnn.params.hidden_states, 1))
            e = 0
            epoch += 1
        textPrev = [rnn.data.char_to_ind[j] for j in rnn.data.text_data[e:e + rnn.params.seq_length]]
        textNext = [rnn.data.char_to_ind[j] for j in rnn.data.text_data[e + 1 : e + rnn.params.seq_length + 1]]
        if smooth_loss <= 40:
            synIndex = rnn.synText(hprev,textPrev[0],1000)
            synText = ''.join(rnn.data.ind_to_char[j] for j in synIndex)
            print('iter %d, loss: %f, epoch: %d' %(i, smooth_loss, epoch))
            print('---------------------------\n %s \n---------------------------' % (synText, ))
        xs, hs, ps, loss = rnn.forwardPass(textPrev,textNext,hprev)
        dW, dU, dV, db, dc, hprev = rnn.backwardPass(xs,hs,ps,textPrev,textNext)
        smooth_loss = smooth_loss *0.999 + loss*0.001
        smoothLosses.append(smooth_loss)
        if i % 10000 == 0:
            print('iter %d, loss: %f, epoch: %d' %(i, smooth_loss, epoch))
        rnn.adagrad(dW, dU, dV, db, dc, mW, mU, mV, mb, mc)
        #rnn.RMSProp(dW, dU, dV, db, dc, mW, mU, mV, mb, mc,0.9)
        #adadelta
        #dWprev, dUprev, dVprev, dbprev, dcprev,delWprev, delUprev, delVprev, delbprev, delcprev =rnn.adadelta(dW, dU, dV,
                 #db, dc, dWprev, dUprev, dVprev, dbprev, dcprev,delWprev, delUprev, delVprev, delbprev, delcprev,gamma=0.9)
        e += rnn.params.seq_length
        i += 1
        if i == 900000:
            plt.plot(range(len(smoothLosses)),smoothLosses)
            plt.xlabel('number of iterations')
            plt.ylabel('smooth loss')
            plt.legend()
            plt.show()
            break



