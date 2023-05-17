import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Bayes theorem (equation 4)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def bayes_theorem(beliefRange,cuePrior,settings):

    margEvid = np.outer((1-beliefRange),cuePrior[0,:])+np.outer(beliefRange,cuePrior[1,:])
    posterior = np.outer(beliefRange,cuePrior[1,:])/margEvid 
    
    return margEvid,posterior

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Calculate the value of termining actions given the belief state (equation 1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def qVal_lr(rewards,beliefState,settings): 
    
    rewards = np.vstack([rewards[0:2],rewards[2:4]])

    qVal_l = np.outer(rewards[0,0],(1-beliefState))+np.outer(rewards[1,0],beliefState)
    qVal_r = np.outer(rewards[0,1],(1-beliefState))+np.outer(rewards[1,1],beliefState)               
    qVal_max = np.max([qVal_l,qVal_r],axis=0)  
    return qVal_l,qVal_r,qVal_max

##So this is an adaptation of Sam's model, because in his you do actually do the right thing
#thus it only effects the value of sampling, another way to edit would be transition probabilities for the terminal state
#then you will just go to the state with the higher expected value for this
#if you might always do the wrong thing, then knowing you are in a place with the best thing doesn't help you
#only if the belief state influences the likely you actually take some action

#Sam's model makes the value of sampling reduce, this seems obvious now...

#the difficult of the question thus is...in what situations would increasing increased state certainity be needed
#that aren't just about actions, where sampling would also increase your action propensity
#there would need to be different actions for each state
#so the actions are zero'd for the other state, but if you select them you terminate
#add this in the email to Sashank

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Calculate the value of taking another sample given the belief state (equation 5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def qVal_cost(costReward,transBelief,beliefValue,cueStep,settings):
        
    qVals_cost = costReward+(np.sum(transBelief*beliefValue))

    return qVals_cost

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Convert values to choice probabilities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def softmax_temperature(qVals,qCosts,inverseTemp):
    
    choiceProbs = np.zeros([qCosts.shape[0],3])
    choiceMade = np.zeros([1,qCosts.shape[0]])
    
    for c in range(qCosts.shape[0]):
                
        values = np.hstack((qVals,qCosts[c]))
        choiceProbs[c,:]=np.round(np.exp(values/inverseTemp)/np.sum(np.exp(values/inverseTemp))*100)/100
        
        #make a choice based on the probabilities
        choiceMade[0,c] = np.array(random.choices([1,2,3], choiceProbs[c,:],k=1))
    
    return choiceProbs,choiceMade

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Return the value of a state based on argmaxing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def belief_values(qValues):

    return np.max([qValues])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Return the value of a state including a pessimistic weight (Zorowitz., 2020)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def belief_values_pessimism(qValues,w):

    findMin = np.min([qValues])
    findMax = np.max([qValues])

    return np.max([qValues])*w + np.min([qValues])*(1-w)
