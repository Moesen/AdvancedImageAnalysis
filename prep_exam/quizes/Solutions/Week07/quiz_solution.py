import numpy as np 
import matplotlib.pyplot as plt
import slgbuilder

I = np.array([[7,1,2,2,4,1],
     [6,6,5,4,5,1],
     [5,2,6,4,4,2],
     [1,5,7,2,2,6],
     [2,4,3,6,7,7]]).astype(np.int32)


r = I.sum(axis=1)

a = r[2]
b = r.min()

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=2, wrap=False)
helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1
c = (I[segmentation_line, np.arange(segmentation.shape[1])]).sum()

fig, ax = plt.subplots()
ax.imshow(I)
ax.plot(segmentation_line, 'r')
ax.set_title(f'cost={c}')

#%% JUST CHECKING
fig, ax = plt.subplots(2,3)
ax = ax.ravel()

for i in range(5):
    ax[i].imshow(I)
    
    delta = i
    
    layer = slgbuilder.GraphObject(I)
    helper = slgbuilder.MaxflowBuilder()
    helper.add_object(layer)
    helper.add_layered_boundary_cost()
    helper.add_layered_smoothness(delta=delta, wrap=False)
    
    helper.solve()
    segmentation = helper.what_segments(layer)
        
    segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1
    cost = (I[segmentation_line, np.arange(segmentation.shape[1])]).sum()
    
    ax[i].imshow(I)
    ax[i].plot(segmentation_line, 'r')
    ax[i].set_title(f'delta={delta}, cost={cost}')

fig.delaxes(ax[5])



