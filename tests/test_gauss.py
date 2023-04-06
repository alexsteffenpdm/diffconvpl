# import the required libraries 
import random 
import matplotlib.pyplot as plt 
    
# store the random numbers in a list 
nums = [] 
mu = 0.0
sigma = 0.05/2
    
for i in range(100): 
    temp = random.gauss(mu, sigma) 
    nums.append(temp) 
        
# plotting a graph 
plt.hist(nums, bins = 100) 
plt.show()