import numpy as np
import scipy.stats as st

class STEPD(object):
    
    def __init__(self, recent_window = 30, alpha_w = 0.05, alpha_d = 0.003):
        
        '''
        
        recent_window = Windowsize of most recent predictions used for drift detection
        alpha_w = Significance level of a warning
        alpha_d = Significance level of a drift
        
        '''
        
        #initialize parameters:
        self.alpha_w = alpha_w
        self.alpha_d = alpha_d
        
        self.r0 = 0
        self.rR = 0
        self.n0 = 0
        self.nR = recent_window
        self.p_hat = None
        self.retrain_memory = []
        self.pred_memory = []
        self.test_statistic = None
        
        self.in_concept_change = None
        self.in_warning_zone = None
        
        #self.reset()
        
        
    def reset(self):
        '''
        reset parameters of change detector
        '''
        self.in_concept_change = False
        self.in_warning_zone = False
        
        self.r0 = 0
        self.rR = 0
        self.n0 = 0
        self.retrain_memory = []
        self.pred_memory = []
        self.test_statistic = None
        self.p_hat = None
        
        # self.__init__(recent_window = self.recent_window, alpha_w = self.alpha_w, alpha_d = self.alpha_d)
        
    
    def add_element(self, prediction):
        '''
        Add new element to the statistic
        
        
        correct classification is indicated with prediction = "1"
        
        '''
        
        if self.in_concept_change:
            self.reset()
        
       
        self.pred_memory.append(prediction)
                    
        #start drift detection if n0 + nR >= 2W
        if len(self.pred_memory[:-self.nR]) >= self.nR:
            self.r0 = sum(self.pred_memory[:-self.nR])
            self.rR = sum(self.pred_memory[-self.nR:])
            self.n0 = len(self.pred_memory[:-self.nR])          
        
            #calculate test statistic:
            self.p_hat = (self.r0 + self.rR) / (self.n0 + self.nR)

            self.test_statistic = np.abs((self.r0/self.n0) - (self.rR/self.nR)) - 0.5*((1/self.n0) + (1/self.nR))
            self.test_statistic = self.test_statistic / np.sqrt(self.p_hat*(1-self.p_hat)*((1/self.n0) + (1/self.nR)))

        
            #get p_value based on gaussian standard normal distribution:
            p_value = 1-st.norm.cdf(abs(self.test_statistic))
            #p_value = st.norm.pdf(abs(self.test_statistic))
                        
                        
            if p_value < self.alpha_w and p_value < self.alpha_d:
                self.in_concept_change = True
        
            elif p_value < self.alpha_w:
                self.in_warning_zone = True
                #append predictions_index in warning zone
                prediction_index = len(self.pred_memory) - 1
                self.retrain_memory.append(prediction_index)
                
            else:
                self.in_warning_zone = False
                self.in_concept_change = False
                
                #remove instances in memory:
                self.retrain_memory = []
        
        
        
    def detected_change(self):
        return self.in_concept_change 



    def detected_warning_zone(self):
        return self.in_warning_zone


    def get_retrain_memory(self):
        #Returns the instances which satisfy p_value < warning level
        return self.retrain_memory
        