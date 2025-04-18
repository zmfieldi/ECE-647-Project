import numpy as np
import matplotlib.pyplot as plt
from gradient import gradient_Dual
from objective import objective

#  Ensure x1, x2 are non-negative
def projection(x, l):
    x1, x2 = x
    l1, l2 = l 
    if x1 < 0 and x2 < 0:
        x1,x2 = 0,0
    elif x1 < 0 and x2 >= 0:
        x1 = 0
        x2 = max((5 + l1 + 2 * l2) / 18, 0)
    elif x2 < 0 and x1 >= 0:
        x2 = 0 
        x1 = max((-2 + 2 * l1 + l2) / 2 , 0)
    return np.array([x1,x2])

#  Ensure lambda is non-negative 
def dual_Projection(l):
    l1, l2 = l
    if l1 < 0:
        l1 = 0
    if l2 < 0:
        l2 = 0
    return np.array([l1,l2])

#  Find optimal x1, x2 based on l1, l2
def optimal(l):
    l1, l2 = l
    x1 = (11 * l1 + 4 * l2 - 17 ) / 9
    x2 = (-4 * l1 + l2 + 16) / 27
    x = projection(np.array([x1,x2]), l)
    return x


#  Gradient Projection Algorithm
def grad_Dual(l_Init, gamma, iters):
    l = np.array(l_Init, dtype=float)
    trajectory = []
    dual_Trajectory = [l.copy()]
    i = 0
    while i < iters:
        x = optimal(l)
        trajectory.append(x.copy())
        gradient = gradient_Dual(x)
        l = l + gamma * gradient
        l = dual_Projection(l)
        dual_Trajectory.append(l.copy())
        i += 1
    trajectory.append(optimal(l))
    return np.array(trajectory), np.array(dual_Trajectory)


#  Start from a random point (scaled to 5) and must be positive
l_Init = np.random.randn(2) 
l_Init *= l_Init
l_Init /= l_Init
l_Init = (l_Init / np.linalg.norm(l_Init) ) * 5
max_iter = 1000
alphas = [0.001, 0.45, 2]  
colors = ['r', 'g', 'b']
labels = ['γ=0.001', 'γ=0.45', 'γ=2']


#  Plot how x changes over k
plt.figure(figsize=(15, 5))
i = 0
for alpha in alphas:
    trajectory, dual_Trajectory = grad_Dual(l_Init, alpha, max_iter)
    k_Vals = np.arange(max_iter + 1)
    plt.subplot(1, 3, i+1)
    plt.plot(k_Vals, trajectory[:,0], color ='r', label="x1")
    plt.plot(k_Vals, trajectory[:,1], color ='y', label="x2")
    plt.plot(k_Vals, dual_Trajectory[:,0], color ='g', label="l1")
    plt.plot(k_Vals, dual_Trajectory[:,1], color ='b', label="l2")
    plt.title(f"X and L Evolving Over k Using {labels[i]}")
    plt.xlabel('$x$')
    plt.ylabel('$k$')
    plt.grid(True)
    plt.legend()
    i += 1
plt.show()
plt.close()

#  Use Objective function for contour plot
x1_Grid = np.arange(-5, 5, .1)
x2_Grid = np.arange(-5, 5, .1)
X1, X2 = np.meshgrid(x1_Grid, x2_Grid)
Z = objective(X1, X2)

#  Plot Trajectory over contour
plt.figure(figsize=(15, 5))
i = 0
for alpha, color, label in zip(alphas, colors, labels):
    trajectory, dual_Trajectory = grad_Dual(l_Init, alpha, max_iter)
    print(F"Final position {trajectory[-1]} on ", label)
    print(F"Final position {dual_Trajectory[-1]} on ", label)
    plt.subplot(1, 3, i+1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, label=label, linewidth=2)
    plt.title('Gradient Descent Trajectories on Contour Plot')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    contours = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)
    i += 1
plt.show()