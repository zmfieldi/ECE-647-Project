import numpy as np
import matplotlib.pyplot as plt
from gradient import grad
from objective import objective

#  Gradient Descent Algo
def grad_Descent(x_Init, gamma, num_iterations):
    x = np.array(x_Init, dtype=float)
    trajectory = [x.copy()]
    i = 0
    while i < num_iterations:
        gradient = grad(x)
        x -= gamma * gradient
        trajectory.append(x.copy())
        i += 1

    return np.array(trajectory)

#  Start from a random point (scaled to 5)
x_Init = np.random.randn(2)
x_Init = (x_Init / np.linalg.norm(x_Init) ) * 5
max_iter = 200
alphas = [0.001, 0.045, 0.13]  
colors = ['r', 'g', 'b']
labels = ['γ=0.001', 'γ=0.045', 'γ=0.13']


#  Plot how x changes over k
plt.figure(figsize=(15, 5))
i = 0
for alpha in alphas:
    trajectory = grad_Descent(x_Init, alpha, max_iter)
    k_Vals = np.arange(max_iter + 1)
    plt.subplot(1, 3, i+1)
    plt.plot(k_Vals, trajectory[:,0], color ='r', label="x1")
    plt.plot(k_Vals, trajectory[:,1], color ='g', label="x2")
    plt.title(f"X Evolving Over k Using {labels[i]}")
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
    trajectory = grad_Descent(x_Init, alpha, max_iter)
    print(F"Final position {trajectory[-1]} on ", label)
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


