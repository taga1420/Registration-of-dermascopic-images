from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pycpd import affine_registration
import numpy as np
import time


def visualize(iteration, error, X, Y, ax):


    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red', s=3, alpha=0.5, edgecolors = 'none')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue',s=3, alpha = 0.5, edgecolors = 'none')
    plt.draw()
    print("iteration %d, error %.5f" % (iteration, error))
    plt.pause(0.001)

    plt.savefig('' + str(iteration) + '.png')

def my_main(X,Y, escala):

    callfun=1
    if callfun==1:
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])

        reg = affine_registration(X, Y, tolerance=0.0001, w=0.0001, escala=escala)
        reg.register(callback)
    #plt.show()
    elif callfun==0:
        reg = affine_registration(X, Y, tolerance=0.0001, w=0.0001, escala=escala)
        reg.register(None)

    return reg

