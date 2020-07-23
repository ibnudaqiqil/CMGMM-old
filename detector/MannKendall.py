import numpy as np
import scipy.stats as st
import pymannkendall as mk

class MannKendall(object):
    
    def __init__(self, min_instances = 30, instances_step = 10, alpha = 0.1, slope_threshold = 0.0, test_type = 'hamed_rao_mod', period = 12):
        
        '''
        
        min_instances = minimum instances to be considered before MK test is applied 
        
        instances_step = after minimum instances is reached, frequency MK test is applied  --> speeds up test significantly if test is not applied every single instance
                         >> "1" = test is applied for every instance
                         >> "10" = test is applied every 10th instance
        
        alpha = Significance level of test
        

        test_type = Type of Test used to perform trend detection:
        
        six different tests available:
        
            - 'original_mk'                    --> Original MK test:  Assumption: No temporal relation in data 
            - 'hamed_rao_mod'                  --> Hamed and Rao Modification MK test:  Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'yue_wang_mod                    --> Yue and Wang Modification MK test:  Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'trend_free_pre_whitening_mod'   --> Trend Free Pre Whitening Modification MK test:  Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'pre_whitening_mod'              --> Pre Whitening Modification MK test:  Assumption: Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'seasonal', period parameter needed! --> Seasonal MK test:  Assumption: temporal relation in data + seasonality 
            
        period = sesonality pattern in dataset -> "12" = monthly, "52" = weekly
        
        '''
        
        #initialize parameters:        
        self.min_instances = min_instances
        self.alpha = alpha
        self.test_type = test_type
        self.period = period
        self.instance_memory = []
        self.slope_threshold = slope_threshold
        self.instances_step = instances_step
        
        self.in_concept_change = False
        
        self.trend = None
        self.p_value = None
        self.sens_slope = 0.0
        self.sample_count = 0
        self.instance_count = 0
        
        
    def reset(self):
        '''
        reset parameters of change detector
        '''
        self.in_concept_change = False        
        self.instance_memory = []
        
        self.trend = None
        self.p_value = None
        self.sens_slope = 0.0
        self.sample_count = 0
        self.instance_count = 0

        # self.__init__(recent_window = self.recent_window, alpha_w = self.alpha_w, alpha_d = self.alpha_d)
        
    
    def add_element(self, value):
        
        '''
        Add new element to the statistic
                
        '''
        
        #reset parameters if change was detected:
        if self.in_concept_change:
            self.reset()
        
        
        
        #append elements:
        self.instance_memory.append(value)
        
                    
        
        if len(self.instance_memory) == self.min_instances:
            self.sample_count = 1
        
        if len(self.instance_memory) > self.min_instances:
            self.instance_count += 1
            
        #start drift detection: >> min_instances have to be reached, then always perform test once, after that perform test every i_th instance (instances_step)
        if len(self.instance_memory) >= self.min_instances and ((self.instance_count == self.instances_step) or (self.sample_count == 1)):
            
            if self.test_type == 'original_mk':
                
                #call corresponding test from package:
                #print('Perform MK test')
                results_tuple = mk.original_test(self.instance_memory, self.alpha)
                #print('MK test ended')
    
            
            if self.test_type == 'hamed_rao_mod':
                
                #call corresponding test from package:
                results_tuple = mk.hamed_rao_modification_test(self.instance_memory, self.alpha)
                
            if self.test_type == 'yue_wang_mod':
                
                #call corresponding test from package:
                results_tuple = mk.yue_wang_modification_test(self.instance_memory, self.alpha)
                
            if self.test_type == 'trend_free_pre_whitening_mod':
                
                #call corresponding test from package:
                results_tuple = mk.trend_free_pre_whitening_modification_test(self.instance_memory, self.alpha)
            
            if self.test_type == 'pre_whitening_mod':
                
                #call corresponding test from package:
                results_tuple = mk.pre_whitening_modification_test(self.instance_memory, self.alpha)
                
            if self.test_type == 'seasonal':
                
                #call corresponding test from package:
                results_tuple = mk.seasonal_test(self.instance_memory, period = self.period, alpha = self.alpha)
            
            
            #reset counter every time a test was performed:
            self.sample_count = 0
            self.instance_count = 0
            
            
            #assign results:
            self.p_value = results_tuple[2]
            self.sens_slope = results_tuple[-1]
            self.trend = results_tuple[0]  
                
                        
            if self.p_value < self.alpha and np.abs(self.sens_slope) > self.slope_threshold:
                self.in_concept_change = True
                   
            else:
                self.in_concept_change = False
    
        
        
        
    def detected_change(self):

        return self.in_concept_change 


    
    def get_test_results(self):
        
        test_results = (self.trend, self.p_value, self.sens_slope)

        return test_results
    