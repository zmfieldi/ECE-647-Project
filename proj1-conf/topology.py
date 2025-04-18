import numpy as np

def get_Topology():
    F = np.zeros((7, 12))    
    # Flow 1
    F[0, 2] = 1  
    F[0, 8] = 1  
    # Flow 2
    F[1, 3] = 1  
    F[1, 8] = 1 
    # Flow 3
    F[2, 2] = 1  
    F[2, 9] = 1  
    # Flow 4
    F[3, 3] = 1  
    F[3, 9] = 1  
    # Flow 5
    F[4, 4] = 1  
    F[4, 7] = 1  
    # Flow 6
    F[5, 4] = 1  
    F[5, 8] = 1  
    # Flow 7
    F[6, 0] = 1
    F[6, 5] = 1 
    F[6, 10] = 1  
    return F
    
