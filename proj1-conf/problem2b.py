import numpy as np
import matplotlib.pyplot as plt
from topology import get_Topology

#  Data and Control Structures 
F = get_Topology()
c = np.array([1,1,1,1,1,2,2,2,2,2,2,2])
w = np.array([1,1,1,1,2,2,2])

#  Ensure x is non-negative and satisfies capacity constraints
def projection(x, l):
    x = np.maximum(0, x)
    return x

#  Ensure lambda is non-negative 
def dual_Projection(l):
    return np.maximum(0, l)

#  Find optimal x based on l
def optimal(l):
    x = np.zeros(7) 
    for i in range(7):
        flow_links = np.where(F[i, :] > 0)[0]
        l_Sum = np.sum(l[flow_links])
        if l_Sum == 0:
            x[i] = 1000  
        else:
            x[i] = w[i] / l_Sum
    
    return x

#  Get the gradient of l
def grad(l):
    x = optimal(l)
    gradient = np.zeros(12)  
    link_Flows = np.zeros(12)
    for i in range(7):  
        for j in range(12):
            if F[i, j] > 0:
                link_Flows[j] += x[i]
    
    gradient = link_Flows - c
    return gradient

#  Gradient Projection Algorithm
def grad_Dual(l_Init, gamma, iters):
    l = np.array(l_Init, dtype=float)
    trajectory = []
    dual_Trajectory = [l.copy()]
    i = 0
    while i < iters:
        x = optimal(l)
        trajectory.append(x.copy())
        gradient = grad(l)
        l = l + gamma * gradient
        l = dual_Projection(l)
        dual_Trajectory.append(l.copy())
        i += 1
    trajectory.append(optimal(l))
    return np.array(trajectory), np.array(dual_Trajectory)

#  Start from a random point (scaled to 5) and must be positive
l_Init = np.random.randn(12) 
l_Init *= l_Init
l_Init /= l_Init
l_Init = (l_Init / np.linalg.norm(l_Init) ) * 5
max_iter = 100
alphas = [0.001, 0.45, 2]  
colors = ['r', 'g', 'b']


#  Plot how x changes over k
plt.figure(figsize=(15, 5))
i = 0
trajectory, dual_Trajectory = grad_Dual(l_Init, .45, max_iter)
k_Vals = np.arange(max_iter + 1)
plt.plot(k_Vals, trajectory[:,0], color ='b', label="x1")
plt.plot(k_Vals, trajectory[:,1], color ='g', label="x2")
plt.plot(k_Vals, trajectory[:,2], color ='m', label="x3")
plt.plot(k_Vals, trajectory[:,3], color ='r', label="x4")
plt.plot(k_Vals, trajectory[:,4], color ='c', label="x5")
plt.plot(k_Vals, trajectory[:,5], color ='y', label="x6")
plt.plot(k_Vals, trajectory[:,6], color ='k', label="x7")
plt.title(f"X Evolving Over k Using γ=0.45")
plt.xlabel('k')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()
plt.close()

#  Plot how l changes over k
trajectory, dual_Trajectory = grad_Dual(l_Init, .45, max_iter)
k_Vals = np.arange(max_iter + 1)
plt.plot(k_Vals, dual_Trajectory[:,0], color ='r', label="l1")
plt.plot(k_Vals, dual_Trajectory[:,1], color ='g', label="l2")
plt.plot(k_Vals, dual_Trajectory[:,2], color ='b', label="l3")
plt.plot(k_Vals, dual_Trajectory[:,3], color ='m', label="l4")
plt.plot(k_Vals, dual_Trajectory[:,4], color ='c', label="l5")
plt.plot(k_Vals, dual_Trajectory[:,5], color ='y', label="l6")
plt.plot(k_Vals, dual_Trajectory[:,6], color ='k', label="l7")
plt.plot(k_Vals, dual_Trajectory[:,7], color ='brown', label="l8")
plt.plot(k_Vals, dual_Trajectory[:,8], color ='olive', label="l9")
plt.plot(k_Vals, dual_Trajectory[:,9], color ='gray', label="l10")
plt.plot(k_Vals, dual_Trajectory[:,10], color ='lime', label="l11")
plt.plot(k_Vals, dual_Trajectory[:,11], color ='gold', label="l12")
plt.title(f"L Evolving Over k Using γ=0.45")
plt.xlabel('k')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()


#  Part C
#  Verify optimality of final flow rates 
def verify_Optimality(x_Final):
    link_Flows = np.zeros(12)
    for i in range(7):
        for j in range(12):
            if F[i, j] > 0:
                link_Flows[j] += x_Final[i]
    
    print("Link capacity constraints:")
    for j in range(12):
        print(f"Link {j+1}: Flow = {link_Flows[j]:.4f}, Capacity = {c[j]}")
    
    marginal_Utils = w / x_Final
    print("\nKKT conditions check:")
    for i in range(7):
        flow_links = np.where(F[i, :] > 0)[0]
        if len(flow_links) > 0:
            print(f"Flow {i+1}: Marginal utility = {marginal_Utils[i]:.4f}")
            print(f"   Used links: {flow_links + 1}")
    
    total_Utility = np.sum(w * np.log(x_Final))
    print(f"\nTotal utility: {total_Utility:.4f}")
    
    return link_Flows, total_Utility


best_alpha = 0.45 
trajectory, dual_Trajectory = grad_Dual(l_Init, best_alpha, max_iter)
x_Final = trajectory[-1]


print("Final flow rates:")
for i in range(7):
    print(f"Flow {i+1}: {x_Final[i]:.4f}")
    
link_Flows, total_Utility = verify_Optimality(x_Final)