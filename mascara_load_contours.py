import cv2
import numpy as np
from examples import fishAffine2D
from examples import fishDeformable2D
from examples import fishRigid2D
import matplotlib.pyplot as plt


image_path1='C:\\Users\\wrt\\Pictures\\IPO_RegistrationSet_BW\\pat1\\1\\P1L1T1.png'

mascara1 = cv2.imread(image_path1, 0)
mascara1_down = cv2.resize(mascara1, None, fx=0.15, fy=0.15)

ret, mascara1_down_thresh = cv2.threshold(mascara1_down, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(mascara1_down_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
height = np.shape(mascara1_down)[0]
width = np.shape(mascara1_down)[1]
result = np.zeros((height, width, 1), np.uint8)

cv2.drawContours(result,contours,-1,(255),1)
# cv2.imshow("Keypoints", result)

mascara1_down = result>200
'''plt.figure()
plt.imshow(mascara1_down)'''
#mascara1_down=result
inds1 = np.argwhere(mascara1_down)
inds1=inds1[:,0:2]
image_path2='C:\\Users\\wrt\\Pictures\\IPO_RegistrationSet_BW\\pat1\\1\\P1L1T2.png'

mascara2 = cv2.imread(image_path2, 0)
mascara2_down = cv2.resize(mascara2, None, fx=0.15, fy=0.15)
ret, mascara2_down_thresh = cv2.threshold(mascara2_down, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(mascara2_down_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
height = np.shape(mascara2_down)[0]
width = np.shape(mascara2_down)[1]
result = np.zeros((height, width, 1), np.uint8)

cv2.drawContours(result,contours,-1,(255),1)
mascara2_down = result>200
inds2 = np.argwhere(mascara2_down)
inds2=inds2[:,0:2]




metodo=1
if metodo==1:
    reg = fishAffine2D.my_main(inds1, inds2, escala=0.15)
    #base (X inicial), rodado (Y final), Nao-rodado (Y inicial)
    #Xmeu, TYmeu, Ymeu = reg.get_info()
    rodado_meu = reg.my_transformPointCloud(np.argwhere(mascara2))
elif metodo==2:
    reg = fishDeformable2D.my_main(inds1, inds2)
    # base (X inicial), rodado (Y final), Nao-rodado (Y inicial)
    Xmeu, TYmeu, Ymeu = reg.get_info()
    rodado_meu = reg.my_transformPointCloud(np.argwhere(mascara2))



print(1)
mascara2_rodada=np.zeros(np.shape(mascara2))

arredondamentos = (rodado_meu+0.5)
arredondamentos=arredondamentos.astype(int)

mascara2_rodada[arredondamentos[:,0], arredondamentos[:,1]]=255


plt.figure()
plt.subplot(2,2,1)
plt.imshow(mascara1-mascara2)

plt.subplot(2,2,2)
plt.imshow(mascara1-mascara2_rodada)
plt.subplot(2,2,3)
plt.imshow(mascara1)
plt.subplot(2,2,4)
plt.imshow(mascara2_rodada)
plt.show()
print(1)
# fishDeformable2D.my_main(inds1, inds2)
