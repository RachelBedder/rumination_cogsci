import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import copy
import importlib

from convenience_functions_rumPOMDP import *

def qvalues_figure(self,rewardID,costID,figSave):

    beliefRange = self.beliefRange

    fig,axs = plt.subplots(1,len(rewardID))
    fig.set_figheight(5)
    fig.set_figwidth(5*len(rewardID))
    lines = [":","dashed","-"]
    lineColors = ["#FEC400","#FEC400","#FEC400"]
    plt.rcParams.update({'font.size': 18})
    
    yLims = [-20,100]

    for r,reward in enumerate(rewardID):
        
        for c,cost in enumerate(costID): #print the threshold lines
            
            qVals = self.actionValues[reward,cost,:,:]
        
            axs[r].plot([self.decisionThreshold[r,c,0],self.decisionThreshold[r,c,0]],yLims,'k',linestyle=lines[c],linewidth=1)
            axs[r].plot([self.decisionThreshold[r,c,1],self.decisionThreshold[r,c,1]],yLims,'k',linestyle=lines[c],linewidth=1)      
        for c,cost in enumerate(costID):  #then print the three costs

            sampleValues = self.actionValues[reward,cost,:,2]
            axs[r].plot(beliefRange,sampleValues,lineColors[c],linestyle=lines[c],linewidth=5) #WILL ONLY PLOT ONE IN FIRST 

        actionValues = self.actionValues[reward,0,:,:]
        axs[0].set_title(self.name)
        axs[r].plot([0,1],[0,0],color = 'k', linestyle = '-')
        axs[r].plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5)
        axs[r].plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5)
        axs[r].set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
        axs[r].set(ylim = yLims)

        axs[r].spines['left'].set_linewidth(3)  
        axs[r].spines['bottom'].set_linewidth(3)
        axs[r].spines['right'].set_linewidth(3)  
        axs[r].spines['top'].set_linewidth(3)  

        if r == 0:
            axs[r].set(ylabel='Value of Action')
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)

        plt.savefig(subPath+'qvalues_'+str(self.name)+'.pdf')
     #   plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')

