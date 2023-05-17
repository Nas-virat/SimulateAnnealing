import math
import random


class SA:
    
    def __init__(self,intialsolution,solutionEvaluator, neighborOperator,initialTemp=300,iterationPerTemp=1000,alpha=0.98,finalTemp=0.01,):
        self.currTemp = initialTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.solution = intialsolution
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        
        self.evaluate = solutionEvaluator
        self.neighborOperator = neighborOperator
        
        self.record = []
        
    def tempReduction(self):
        self.currTemp = self.currTemp * self.alpha
        
    def isTerminationCriteriaMet(self):
        return self.currTemp <= self.finalTemp
    
    def run(self):
        while not self.isTerminationCriteriaMet():
           
            for i in range(self.iterationPerTemp):
               
                neighbor = self.neighborOperator(self.solution)
                
                cost = self.evaluate(neighbor) - self.evaluate(self.solution) 
                
                # new solution is better
                if cost < 0:
                    self.solution = neighbor
                else:
                    if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                        self.solution = neighbor
                        
                    
        return self.solution, self.evaluate(self.solution),self.record
                
    
    
    
    
        
        
       
    
    