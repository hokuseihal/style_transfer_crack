import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from sklearn.metrics import average_precision_score as map
from sklearn.metrics import roc_auc_score as auc
plt.gray()

def showimg(img,target,value=None):
    plt.subplot(1,2,1)
    if value:
        plt.title(f'{value:.3f}')
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(target)
    plt.show()


folder='/home/hokusei/Pictures/0403_cut/'
acc_map=np.zeros((2,2))
bi_thresh =lambda img:cv2.threshold(img,0,255,cv2.THRESH_OTSU)[1]
bilateral=lambda img:cv2.bilateralFilter(img,-1, 20,10)
nlm=lambda img: cv2.fastNlMeansDenoising(img,None,20,7,21)
bilateral_2=lambda img:bilateral(bilateral(img))
bilateral_3=lambda img:bilateral_2(img)
ident=lambda img:img
simple_thresh=lambda img:img<80

for method_idx,f in enumerate([ident,bilateral,bilateral_2,bilateral_3,nlm]):
    mean=[]
    flag=[]
    for fol in ['clear_negative','clear_positive','shade_negative','joint']:
        for img_p in glob.glob(folder+fol+'/*.jpg'):
            im=cv2.imread(img_p,0)
            #if fol=='joint':
            #    showimg(im,simple_thresh(f(im)),simple_thresh(f(im)).mean())
            im=simple_thresh(f(im)).mean()
            mean.append(im)
            flag.append((fol in ['joint','clear_positive'])*1)
        print(f'{fol} {np.mean(mean)}')
    mean=np.array(mean)
    flag=np.array(flag)
    print(f'map={map(flag,mean/mean.max())}')
    print(f'auc={auc(flag,mean/mean.max())}')
    plt.hist([mean[0:100],mean[100:200],mean[200:300],mean[300:400]],stacked=False,color=['red','blue','m','c'],alpha=0.5,bins=50)
    plt.xlim(0,0.1)
    plt.show()