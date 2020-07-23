import numpy as np
import math

class HDDDM(object):
    
    '''
    Implementation of the Drift Detection algorithm based on Hellinger Distance of Histograms (Ditzler, Polikar 2011)
    '''
    
    def __init__(self, batch_size = 30, gamma = 1.5):
        
        '''
        batch_size = number of instances in current batch presented to the algorithm

        
        '''
        
        #initialize parameters:
        self.batch_size = batch_size     
        self.hist_P = []
        self.hist_Q = []
        self.instance_memory = []
                
        self.init_flag = False
        
        self.hellinger_t = None
        self.hellinger_t_1 = 0
        self.epsilon = None
        self.epsilon_memory = []
        self.gamma = gamma
    
        self.in_concept_change = None
        
        
    def reset(self):
        '''
        reset parameters of change detector
        '''
        self.in_concept_change = False
        
        self.epsilon = None    
        #only keep epsilon of last detected change:
        self.epsilon_memory = [self.epsilon_memory[-1]]
                
        self.hist_Q = self.hist_P
        self.hist_P = []
        

    
    def discrete_hellinger_distance(self, p,q):
        
        list_of_squares = []
        for p_i, q_i in zip(p,q):
            
            #square of difference of ith element:
            s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2
            
            #append
            list_of_squares.append(s)
    
        #get sum of squares:
        sum_sq = sum(list_of_squares)
        
        return math.sqrt(sum_sq/2)
    
    
    def hist_hellinger_distance(self, hist_Q, hist_P, n_bins):

        freqs_memory_Q = []
        freqs_memory_P = []

        hist_result_Q = np.histogram(hist_Q, bins=n_bins)
        hist_result_P = np.histogram(hist_P, bins=n_bins)

        hist_sum_Q = sum(hist_result_Q[0])
        hist_sum_P = sum(hist_result_P[0])


        for i in range(len(hist_result_Q[0])):

            hist_freq_Q = np.sqrt((hist_result_Q[0][i]/hist_sum_Q))
            hist_freq_P = np.sqrt((hist_result_P[0][i]/hist_sum_P))

            freqs_memory_Q.append(hist_freq_Q)
            freqs_memory_P.append(hist_freq_P)


        hell_distance = np.sqrt(np.sum((np.array(freqs_memory_P) - np.array(freqs_memory_Q))**2))


        return hell_distance

            
    
    def add_element(self, value):
        '''
        Add new element to the statistic and create batches.
        If batch size is reached, compute Hellinger Distance based on Ditzler and Polikar 2011
                
        '''
        
        if self.in_concept_change:
            self.reset()
        
        
        #append new instances 
        self.instance_memory.append(value)      
        
        #initialize hist_Q in first iteration:
        if len(self.instance_memory) == self.batch_size and not self.init_flag:
            self.hist_Q = self.instance_memory
            
            #empty list:
            self.instance_memory = []
            
            self.init_flag = True
           
        
        #initialize hist_P:
        if len(self.instance_memory) == self.batch_size and self.init_flag:
            self.hist_P = self.instance_memory
            
            #empty list:
            self.instance_memory = []
            
            

        if len(self.hist_P) == self.batch_size:
            
        
            #compute Hellinger Distance:
            n_bins = math.floor(np.sqrt(len(self.hist_P))) #based on cardinality of self.hist_P
            self.hellinger_t = self.hist_hellinger_distance(self.hist_Q, self.hist_P, n_bins)

            #compute measures to update threshold:
            mean_eps = np.mean(np.abs(self.epsilon_memory))
            std_eps = np.std(np.abs(self.epsilon_memory))

            # get difference in divergence:
            self.epsilon = self.hellinger_t - self.hellinger_t_1

            # append epsilon:
            self.epsilon_memory.append(self.epsilon)

            #compute threshold:
            beta_t = mean_eps + self.gamma*std_eps
                       
            #update hellinger distance at (t-1)
            self.hellinger_t_1 = self.hellinger_t

            
            if abs(self.epsilon) > beta_t:
                self.in_concept_change = True

            else:
                #update hist_Q:
                self.hist_Q = self.hist_Q + self.hist_P 
                #empty self.hist_P:
                self.hist_P = []
                    
        
        
    def detected_change(self):

        return self.in_concept_change 
