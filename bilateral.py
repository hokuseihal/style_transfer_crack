import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.gray()
img = cv2.imread('pic1.jpg',0)

plt.subplot(3,4,1)
plt.title('origin')
plt.imshow(img)

blur = cv2.blur(img,(10,10))
plt.subplot(3,4,2)
plt.title('box filter')
plt.imshow(blur)
plt.subplot(3,4,3)
plt.title('bilateral')
bil=cv2.bilateralFilter(img,-1, 50, 10)
plt.imshow(bil)
plt.subplot(3,4,4)
plt.title('non local mean')
nlm=cv2.fastNlMeansDenoising(img,None,50,7,21)
plt.imshow(nlm)

plt.subplot(3,4,5)
bw=[img<128][0]*256
plt.imshow(bw)

plt.subplot(3,4,6)
blurbw=[blur<128][0]*256
plt.imshow(blurbw)

plt.subplot(3,4,7)
bilbw=[bil<128][0]*256
plt.imshow(bilbw)

plt.subplot(3,4,8)
nlmbw=[nlm<128][0]*256
plt.imshow(nlmbw)

kernel = np.ones((5,5),np.uint8)

plt.subplot(3,4,9)
plt.imshow(cv2.morphologyEx(bw.astype(np.float32),cv2.MORPH_OPEN,kernel))
print(bw.astype(np.uint8).max())
plt.subplot(3,4,10)
plt.imshow(cv2.morphologyEx(blurbw.astype(np.float32),cv2.MORPH_OPEN,kernel))

plt.subplot(3,4,11)
plt.imshow(cv2.morphologyEx(bilbw.astype(np.float32),cv2.MORPH_OPEN,kernel))

plt.subplot(3,4,12)
plt.imshow(cv2.morphologyEx(nlmbw.astype(np.float32),cv2.MORPH_OPEN,kernel))



plt.show()
