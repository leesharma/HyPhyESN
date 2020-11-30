import numpy as np
import networkx as nx

class CHyPPESN():

    def __init__(self, n_outputs, n_reservoir=200, bias=0,
                 spectral_radius=0.95, sparsity=0, fb_spectral_radius=0.95, fb_sparsity=0, noise=0.001, leak=1,lmbda = 0.1,
                 square = False,
                 silent=True,random_state=None):

        # initial variable
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.bias=bias
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.fb_spectral_radius = fb_spectral_radius
        self.fb_sparsity = fb_sparsity
        self.noise = noise
        self.leak=leak
        self.square = square
        self.silent=silent
        self.random_state = random_state
        self.lmbda = lmbda

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.initweights()

    # initial weight
    def initweights(self):
        # initialize reservoir weights:
        # (random matrix method)
        # begin with a random matrix:
        # W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # remove diagonal
        # np.fill_diagonal(W, 0)
        # delete the fraction of connections given by sparsity:
        # W[self.random_state_.rand(*W.shape) < self.sparsity] = 0

        # (random graph method, try to find a better way to use degree of graph instead of the sparsity).
        G = nx.fast_gnp_random_graph(self.n_reservoir,1-self.sparsity,directed=True) #generate random graph
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = self.random_state_.rand() - 0.5 #assign each edge a random number
        W = nx.to_numpy_matrix(G) #change graph to matrix
        
        # Set the correct spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)

        # Generate W_in:
        W_in = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1
        # delete the fraction of connections given by sparsity:
        W_in[self.random_state_.rand(*W_in.shape) < self.fb_sparsity] = 0
        #Set the correct spectral radius
        fb_radius = np.max(np.abs(np.sqrt(np.linalg.eigvals(np.dot(W_in.T, W_in)))))
        self.W_in = W_in * (self.fb_spectral_radius / fb_radius)

    # update function for the reservoir
    def _update(self, state, outputs):
        prep = (np.dot(self.W, state)+ np.dot(self.W_in, outputs)+self.bias) #Reservoir update in time
        return ((1-self.leak)*state+ self.leak*np.tanh(prep)
            + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    # for ridge regression
    def set_lmda(self, lmbda=0.1):
        self.lmbda = lmbda
        return self

    def fit(self, outputs, target, func):
        
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        states1 = np.zeros((outputs.shape[0], self.n_reservoir))
        hybrid_term = np.zeros((outputs.shape[0], self.n_outputs))
        for n in range(1, outputs.shape[0]):
            states1[n, :] = self._update(states1[n - 1], outputs[n - 1])
            hybrid_term[n, :] = func(outputs[n - 1])

        discard = min(int(outputs.shape[1] / 10), 100)
        

        if self.square:
            states2 = np.square(states1)
            states = np.stack((states1,states2),axis=1).reshape((outputs.shape[0], 2*self.n_reservoir))
        else:
            states = states1
            
        ### reservoir states + hybrid constraints
        print(states.shape)
        print(hybrid_term.shape)
        states = np.hstack((states,hybrid_term))

        # compute W_out (ridge regression)
        C = states[discard:, :].T.dot(states[discard:, :]) + self.lmbda*np.eye(states[discard:, :].shape[1])
        self.W_out = np.linalg.inv(C).dot(states[discard:, :].T.dot(target[discard:, :]))
        # self.W_out = np.dot(np.linalg.pinv(states[discard:, :]),outputs[discard:, :])

        self.laststate = states[-1, 0:self.n_reservoir]
        self.lastoutput = target[-1, :]

        pred_train = np.dot(states, self.W_out)
        if not self.silent:
            print("training error:")
            print(np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train

    def predictnext(self, output, func, continuation=True):
        
        #only make one step of prediction, make loop outside.

        laststate = self.laststate
        states = np.vstack([laststate, np.zeros(self.n_reservoir)])
        states[1,:] = self._update(states[0,:] ,output)
        if self.square:
             prep=np.reshape(np.hstack((states[1,:], np.square(states[1,:]))),-1)
        else:
             prep = states[1,:]
                
        hybrid_term = func(output)
        prep = np.hstack((prep,hybrid_term))
             
        nextout = np.dot(self.W_out.T,prep)
        self.laststate = states[1,:]
        self.lastoutput = nextout
        
        return nextout

