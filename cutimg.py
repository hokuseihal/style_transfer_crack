import matplotlib.pyplot as plt
import numpy as np
import glob
import os
def subdir(path,dir):
    t=path.split('/')
    t.insert(-1,dir)
    return '/'.join(t)


imgs='/home/hokusei/Videos/frame/'
fol='cut'
os.makedirs(imgs.replace('frame',fol),exist_ok=True)
for imgp in glob.glob(imgs+'*.jpg'):
    im=plt.imread(imgp)
    im=im[2346:4096,205:1955]
    plt.imsave(imgp.replace('frame','cut'),im)
