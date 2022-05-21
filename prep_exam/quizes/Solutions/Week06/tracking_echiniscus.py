import imageio
import numpy as np
import matplotlib.pyplot as plt
import simple_snake as sis

filename = '../../../../Data/week6/echiniscus.mp4';
vid = imageio.get_reader(filename)
movie = np.array([im for im in vid.iter_data()], dtype=np.float)/255
gray = (2*movie[:,:,:,2] - movie[:,:,:,1]- movie[:,:,:,0]+2)/4


#%% settings
nr_points = 100
step_size = 10
alpha = 0.1
beta = 0.1
center = np.array([120,200])
radius = 40
start_frame = 74

#%% initialization
snake = sis.make_circular_snake(nr_points, center, radius)
B = sis.regularization_matrix(nr_points, alpha, beta)
g = gray[start_frame] # image we operate on
m = movie[start_frame] # image we display
fig, ax = plt.subplots()
ax.imshow(m, cmap='gray')
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')

for i in range(50):    
    snake = sis.evolve_snake(snake, g, B, step_size)    
    ax.clear()
    ax.imshow(m, cmap='gray')
    ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
    ax.set_title(f'initialization, iter {i}')
    plt.pause(0.001)
  
      
#%% tracking
for i in range(0,300):
    m = movie[start_frame + i] # 2 evolution steps per frame
    g = gray[start_frame + i]
    snake = sis.evolve_snake(snake, g, B, step_size)    
    ax.clear()
    ax.imshow(m, cmap='gray')
    ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
    ax.set_title(f'tracking, frame {i}')
    plt.pause(0.001)






