import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
import skimage.morphology as morph
import numpy.random
import cv2
import time

print("Hello world")

IMDIR=r"C:\Users\flole\Desktop\TP5\catheter"



# #Affichage de toutes les images
# im_counter=1
# for root, dirnames, filenames in os.walk(IMDIR):
#     for filename in filenames:
#         f=os.path.join(IMDIR, filename)
#         if f.endswith(".png"):
#             img=cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#             plt.subplot(2,5,im_counter)
#             plt.imshow(img, cmap='gray')
#             plt.grid(False)
#             plt.axis('off')
#             im_counter += 1
# plt.show()
#on ouvre les images et on les convertie en floats pour faire l'opération de soustraction

ref=cv2.imread(r"C:\Users\flole\Desktop\TP5\catheter\frame_201.png", cv2.IMREAD_GRAYSCALE)
ref=np.float32(ref)
img=cv2.imread(r"C:\Users\flole\Desktop\TP5\catheter\frame_291.png", cv2.IMREAD_GRAYSCALE)
img=np.float32(img)

img=img-ref
#on ramène l'image sur 8 bits
img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# debruitée=cv2.bilateralFilter(img, 5, 20, 20)
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(debruitée, cmap='gray')
# plt.show()
# #cv2.imshow("Debruitee", debruitée)

img_gauss=cv2.GaussianBlur(img, (3,3), 1)
plt.imshow(img_gauss, cmap='gray')
plt.show()

im_bil=cv2.bilateralFilter(img_gauss, 5, 20, 20)
plt.imshow(im_bil, cmap='gray')
plt.show()

#on retire sa moyenne à l'image pour avoir les motifs en noir sur du blanc
centrée=img_gauss-np.mean(img_gauss)
plt.imshow(centrée, cmap='gray')        
plt.show()

#On vire tout le bruit de fond en mettant à zéro les pixels négatifs
centrée[centrée<0]=0

plt.imshow(centrée, cmap='gray')
plt.title("Image centrée et bruit de fond supprimé")
plt.show()

#on érode l'image pour ne garder que les gros traits
erode=cv2.erode(centrée, np.ones((3,3), np.uint8), iterations=1)
plt.imshow(erode, cmap='gray')
plt.title("Image érodée")
plt.show()

#filtrage de Canny-Edges en essayant plein de combinaisons de seuils

edges=cv2.Canny(np.uint8(centrée), 100, 150)
plt.imshow(edges, cmap='gray')
plt.title("Canny-Edges avec seuils 50 et 100")
plt.show()

from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import skeletonize

#On épaissit les traits de Kennys pour que la détection de contours soit plus facile
epaissie=cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
plt.imshow(epaissie, cmap='gray')
plt.show()

#puis on éroder pour ne garder que les plus gros traits
erode=cv2.erode(epaissie, np.ones((4,4), np.uint8), iterations=1)
plt.imshow(erode, cmap='gray')
plt.show()
