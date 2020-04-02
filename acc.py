import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.gray()

msef=lambda x,y: ((x-y)**2).mean()
def showimg(img,target,thresh=None):
    plt.subplot(1,2,1)
    if thresh:
        img=img<thresh
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(target)
    plt.show()


folder='crack/img/'
acc_map=np.zeros((2,2))
bi_thresh =lambda x:x
bilateral=lambda img:cv2.bilateralFilter(img,-1, 50, 10)
nlm=lambda img: cv2.fastNlMeansDenoising(img,None,50,7,21)

for f in [bilateral,nlm]:
    for thresh in np.linspace(0,1):
        for img_p in glob.glob(folder+'/**/*.jpg'):
            out=(f(cv2.imread(img_p,0)).mean()>thresh)*1
            target=('positive' in img_p)*1

            acc_map[target,out]+=1
            
        print(f'{f=} {thresh=} {acc_map=}')
