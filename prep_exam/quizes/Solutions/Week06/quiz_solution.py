import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import simple_snake as sis
import skimage.draw

a = (40+10*1)**2*np.pi
print(a)

filename = '../../../../Data/week6/plusplus.png'
I = skimage.io.imread(filename).astype(np.float)
I = np.mean(I,axis=2)/255

snake = sis.make_circular_snake(200, np.array(I.shape)/2, 180)

mask = skimage.draw.polygon2mask(I.shape, snake.T)
m = np.array([np.mean(I[~mask]), np.mean(I[mask])])

b = m[1]
print(b)

J = m[mask.astype(int)]

e = (I-J)**2
c = e.sum()
print(c)


fig, ax = plt.subplots(1,3)
ax[0].imshow(I, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')

ax[1].imshow(J, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_title(f'mean_in = {b:0.3f}')
ax[2].imshow(e, cmap=plt.cm.gray)
ax[2].set_title(f'external_energy = {c:0.1f}')




