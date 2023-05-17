#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### HOUSEKEEPING
#import what is necessary and some things that may be unnecessary, and convenience functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import copy
import importlib

from modelFunctions_rumPOMDP import *
from convenience_functions_rumPOMDP import *

class POMDP:
    
    def __init__(self,xObvs,obvStep):
        
        #...set the parameters for the observation distributions
        self.name = xObvs[0]
        self.xMeans = np.array(xObvs[1:3])
        self.xStd = np.array(xObvs[3:5])
        self.obvs = np.arange(obvStep,100,obvStep) 
        
        stateNo = self.xMeans.shape[0]
        obvPrior = np.zeros([stateNo,len(self.obvs)]) #...priors for all observations (typically set at 0)
        
        beliefD = 0.01 #...discretize the belief space
        beliefRange = np.arange(0,1+beliefD,beliefD) #...generate all possible values of b, given the discretization
        
        for x in range(stateNo): 

            obvMean = self.xMeans[x]
            obvStd = self.xStd[x]

            obvPrior[x,:] = stats.norm.pdf(self.obvs,obvMean,obvStd)*obvStep
           
        self.margEvid,self.posterior = bayes_theorem(beliefRange,obvPrior,'-')
        
        obvPrior[obvPrior<0.001] = 0.001#...do not allow non-zero observation priors
        
        #...normalize each row so the sum of belief states is 1
        row_sums = obvPrior.sum(axis=1)
        obvPrior = obvPrior / row_sums[:, np.newaxis]
        
        self.beliefRange = beliefRange
        self.obvPrior = obvPrior
        self.obvStep = obvStep

        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### calculate the belief transition matrix
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
    def belief_transition_matrix(self):
        
        self.beliefTrans = np.zeros([len(self.beliefRange),len(self.beliefRange)])
        
        for b,belief in enumerate(self.beliefRange):
            
            beliefTrans = self.beliefTrans[b,:] #...transition vector for P(B'|B_i)

            _,bPrimeRange = bayes_theorem(belief,self.obvPrior,'-') #...for each belief and each possible cue, find all probabilities
            bPrimeRange = bPrimeRange[0]
            
            for bp,bPrime in enumerate(bPrimeRange): #...tmp is a vector of all possible B' for B_i
                                
                bFind = index_finder(self.beliefRange,bPrime) #...returns the closest B to the B' being iterated through
                
                beliefTrans[bFind] = beliefTrans[bFind]+self.margEvid[b,bp] 
               
            self.beliefTrans[b,:] = beliefTrans
              
    def belief_transition_matrix_plot(self,figSave):
        
            fig, ax = plt.subplots(1,2)
            fig.set_figheight(5)
            fig.set_figwidth(5*3)

            ax[0].plot(self.obvs,np.transpose(self.obvPrior),linewidth=8)
            ax[0].set(ylim=[0,0.1],xlim=[0,100],title=self.name,xlabel='Observation estimate', ylabel='Probability')
            ax[0].set_aspect('equal', adjustable='box')
            ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')

            beliefMatrix = self.beliefTrans

            im = ax[1].imshow(self.beliefTrans, cmap='copper', aspect = 'auto',vmin = 0,vmax = 0.04)
            ax[1].set(xlabel='B\'',ylabel='B')
            fig.colorbar(im, ax=ax[1], label='P(B\'|B)')
            ax[1].set_aspect(1.0/ax[1].get_data_ratio(), adjustable='box')
            plt.tight_layout()

            if figSave != "": #...print individual to file

                # create the full path for the subfolder
                subPath = os.path.join(figSave, "observation_distributions/")

                # create the subfolder if it does not exist
                if not os.path.exists(subPath):
                    os.makedirs(subPath)

                plt.savefig(subPath+'belief_transition'+str(self.name)+'.pdf')
          #      plt.savefig(subPath+'belief_transition'+str(self.name)+'.svg')

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## calculate the value of each state using dynamic programming
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def value_iteration(self,allRewards,allCosts,figPrint,figSave):
        
        beliefRange = self.beliefRange
        beliefTrans = self.beliefTrans
        
        self.beliefValues = np.full([allRewards.shape[0],allCosts.size,len(beliefRange)],np.nan)
        self.actionValues = np.full([allRewards.shape[0],allCosts.size,len(beliefRange),3],np.nan)
        self.decisionThreshold = np.full([allRewards.shape[0],allCosts.size,2],np.nan)
        
        self.allRewards = allRewards
        self.allCosts = allCosts
        
        if figPrint :
            fig, ax = plt.subplots(2,allRewards.shape[0])
            fig.set_figheight(5*2)
            fig.set_figwidth((5*allRewards.shape[0]))

        for r,rewards in enumerate(allRewards):
            
            for c,cost in enumerate(allCosts):
        
                beliefValues = np.zeros([len(beliefRange)])
                actionValues = np.zeros([len(beliefRange),3])
                
                vDelta = 1 #...threshold for adjusting the value in an iteration
                qDelta = 1 #...RB
                
                updateTrue = True
                
                plotValue = []
                plotAction = []
                k = 0 #...this will iterate as the values converge to within the vDelta amount
                
                while updateTrue:    
                    
                    v = copy.copy(beliefValues) #...vector of all the belief values which is iterated on each k

                    for b,belief in enumerate(beliefRange): #...for each belief

                        bPrime = beliefTrans[b,:] #...vector of state transition pro babilities for some belief state
                        
                        #...functions for calculating all q values
                        qSample = qVal_cost(cost,bPrime,v,self.obvStep,'-') 
                        qL,qR,_ = qVal_lr(rewards,belief,'-') 
                      
                        actionValues[b,:] = [qL,qR,qSample]
                        
                        beliefValues[b] = belief_values(actionValues[b,:])

                    #...when the maximum difference between any of the belief values on each iterate is more than vDelta,
                    #then continue to iterate until this is not true.
                    updateTrue = np.max(abs(v-beliefValues))>vDelta 
                    
                    plotValue = np.append(plotValue,beliefValues)
                    plotAction = np.append(plotAction,actionValues[:,2])
                    
                    k = k+1 #...keep counting the iterations!
                    
                    #...save the final belief values and action values
                    self.beliefValues[r,c,:] = beliefValues
                    self.actionValues[r,c,:,:] = actionValues
                    
                    #...save the decision thresholds for each reward function and cost
                    self.decisionThreshold[r,c,0] = beliefRange[index_finder(actionValues[:,2],actionValues[:,0])]
                    self.decisionThreshold[r,c,1] = beliefRange[index_finder(actionValues[:,2],actionValues[:,1])]  
                    
                    if figPrint:

                        plotValue = np.resize(np.transpose(plotValue),(k,len(beliefRange)))
                        plotAction = np.resize(np.transpose(plotAction),(k,len(beliefRange)))

                        im = ax[0,r].imshow(plotValue, cmap='gray', aspect = 'auto')
                        im = ax[1,r].imshow(plotAction, cmap='gray', aspect = 'auto')
                        if r == 0:
                            im = ax[0,r].set_title(self.name)

                        ax[0,r].set(xlim = [0,100],xticks = [0,0.5,1],xlabel ='Belief State $B$',ylabel='k for V(B)')
                        ax[1,r].set(xlim = [0,100],xticks = [0,0.5,1],xlabel ='Belief State $B$',ylabel='k for Q(Sample)')

                if (figSave != "") & (c==0): #...print individual to file

                    # create the full path for the subfolder
                    subPath = os.path.join(figSave, "value_iteration/")

                    # check if the subfolder already exists
                    if not os.path.exists(subPath):
                    # create the subfolder if it does not exist
                        os.makedirs(subPath)

                    plt.savefig(subPath+'value_iteration'+str(self.name)+'.pdf')
             #       plt.savefig(subPath+'value_iteration'+str(self.name)+'.svg')


        
