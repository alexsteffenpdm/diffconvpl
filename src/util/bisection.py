import numpy as np
from tqdm import tqdm
import torch


    


if __name__ == "__main__":
    domain = np.linspace(-1.0,1.0,5)
    sq_domain = np.multiply(domain,domain)
    # points = np.asanyarray([ [np.random.uniform(-1.0,1.0),np.random.uniform(-10.0,10.0)] for _ in range(1)])
    points = np.asanyarray([ [0.0,1.0] for _ in range(1)])
    matrix = np.zeros((len(points),len(domain),4))

    for i in tqdm(range(matrix.shape[0])): # 1000
        for j in range(matrix.shape[1]): # 20000
            x = domain[j]
            y = sq_domain[j]
            px = points[i][0]
            py = points[i][1]
            matrix[i][j] = np.array([x,y,px,py])

    t = torch.from_numpy(np.asanyarray(matrix)).type(torch.FloatTensor)
   
    print(matrix)
    funcT = lambda p: torch.sqrt( torch.pow(p[0] - p[2],2) + torch.pow(p[1] - p[3],2))
    funcN = lambda p: np.sqrt( (p[0] - p[2])**2 + (p[1] - p[3])**2)
    t1 = funcT(t)
    m1 = funcN(matrix)
    print(np.min(m1),torch.min(t1))
    #print(np.min(m1,dim=1),torch.min(t1,dim=1))
    print(np.min(m1),torch.min(t1,dim=-1))
    
    
    



