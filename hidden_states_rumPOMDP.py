# Script to generate the different parameters for hidden states
import matplotlib.pyplot as plt
import numpy as np
        
def make_states(std_x1,std_x2,mean_x1,mean_x2): 
    
    idList = range(len(mean_x1))

    xObvs = []
    idList = 0

    for (std1,std2) in zip(std_x1,std_x2): 

        for (m1,m2) in zip(mean_x1,mean_x2):

            simName = str(idList)+"_md"+str(m2-m1)+"_std1_"+str(std1)+"_std2_"+str(std2)
            xObvs.append([simName,m1,m2,std1,std2])
            idList = idList+1
                
    return xObvs

def build_hidden_states(whatState):
    
    print(whatState)
    
    if whatState == "baseline":
        std_min = 10
        std_max = 20
        std_step = 5
        std_max +=1 #so it uses the max
        std_range = np.arange(std_min,std_max,std_step)
        std_range[std_range==0] = 1 #...std of 0 will crash

        std_x1 = std_range 
        std_x2 = std_range

        meanDist_min = 10
        meanDist_max = 20
        meanDist_step = 5 
        meanDist_range = np.arange(meanDist_min,meanDist_max,meanDist_step)/2

        meanDist_start = 50 #...this is where the means will be when the meanDist is 0

        mean_x1 = list(meanDist_start-meanDist_range)
        mean_x2 = list(meanDist_start+meanDist_range)

    if whatState == "basic":
        std_min = 5
        std_max = 50
        std_step = 5
        std_max +=1 #so it uses the max
        std_range = np.arange(std_min,std_max,std_step)
        std_range[std_range==0] = 1 #...std of 0 will crash

        std_x1 = std_range 
        std_x2 = std_range

        meanDist_min = 5
        meanDist_max = 50
        meanDist_step = 5 
        meanDist_range = np.arange(meanDist_min,meanDist_max,meanDist_step)/2

        meanDist_start = 50 #...this is where the means will be when the meanDist is 0

        mean_x1 = list(meanDist_start-meanDist_range)
        mean_x2 = list(meanDist_start+meanDist_range)
        
    if whatState == "test":

        std_min = 20
        std_max = 30
        std_step = 5
        std_max +=1 #so it uses the max
        std_range = np.arange(std_min,std_max,std_step)
        std_range[std_range==0] = 1 #...std of 0 will crash

        std_x1 = std_range 
        std_x2 = std_range

        meanDist_min = 10
        meanDist_max = 20
        meanDist_step = 5 
        meanDist_range = np.arange(meanDist_min,meanDist_max,meanDist_step)/2

        meanDist_start = 50 #...this is where the means will be when the meanDist is 0

        mean_x1 = list(meanDist_start-meanDist_range)
        mean_x2 = list(meanDist_start+meanDist_range)
    
    xObvs = make_states(std_x1,std_x2,mean_x1,mean_x2) 
        
    return  xObvs      
        
