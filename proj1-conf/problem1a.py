from objective import objective
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#  3D graph
ax = plt.axes(projection="3d")

x1_Grid = np.arange(-5, 5, .1)
x2_Grid = np.arange(-5, 5, .1)
X1, X2 = np.meshgrid(x1_Grid, x2_Grid)

Z = objective(X1,X2)

#  Plot Formatting
surface = ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('3D Surface Plot of f(x1,x2)')



#  2D plots 
x0 = np.array([0, 0])
plt.figure(figsize=(15, 5))

#  Loop for 3 random Directions
for i in range(3):
    #  Random Unit Vector
    direction = np.random.randn(2)
    direction = direction / np.linalg.norm(direction) 
    
    t_values = np.arange(-5, 5, .1)

    #  Calculate f(t) 
    function_values = []
    for t in t_values:
        point = x0 + t * direction
        function_values.append(objective(point[0], point[1]))
    
    #Plot Formatting
    plt.subplot(1, 3, i+1)
    plt.plot(t_values, function_values)
    plt.grid(True)
    plt.title(f'Direction {i+1}: [{direction[0]:.2f}, {direction[1]:.2f}]')
    plt.xlabel('t')
    plt.ylabel('f(t)')
plt.show()
