import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import simple_snake as sis

filename = '../../../../Data/week6/plusplus.png'
I = skimage.io.imread(filename).astype(np.float)
I = np.mean(I,axis=2)/255

nr_points = 100
nr_iter = 100
step_size = 5
alpha = 0.01
beta = 0.1

center = np.array(I.shape)/2
radius = 0.3*np.mean(I.shape)

snake = sis.make_circular_snake(nr_points, center, radius)
B = sis.regularization_matrix(nr_points, alpha, beta)

fig, ax = plt.subplots()
ax.imshow(I, cmap=plt.cm.gray)
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
ax.set_title('Initialization')

for i in range(nr_iter):
    snake = sis.evolve_snake(snake, I, B, step_size)
    ax.clear()
    ax.imshow(I, cmap=plt.cm.gray)
    ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
    ax.set_title(f'iteration {i}')
    plt.pause(0.001)

      












