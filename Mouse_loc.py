import matplotlib.pyplot as plt
import numpy as np

im = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/extra/20.tif')
ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(im)
loc = []

def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
        loc.append(event.xdata)
        loc.append(event.ydata)
        

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()