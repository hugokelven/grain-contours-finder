#pip: 19.0.3
#pillow: 8.4.0
#opencv-python: 4.5.4.60
#numpy: 1.21.4
#matplotlib: 3.5.0

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imagens/metallographic_01.jpg')

gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) #converte a escala de cores da imagem para a escala cinza
gray = cv2.GaussianBlur(gray, (5, 5), 0) #aplica um filtro que borra a imagem

lower = 125
upper = 150

edges = cv2.Canny(gray, lower, upper) #aplica o algoritmo de detecção de bordas
edges = cv2.dilate(edges, None, iterations=1) #dilata os contornos para uní-los
edges = cv2.erode(edges, None, iterations=1) #disfaz o dilate, porém mantém os contornos unidos

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #detecta os contornos

length = len(contours)
minarea = 25 #defini uma área mínima para ser considerada um grão
cnt = []
areas = []

for i in range(length):
    area = cv2.contourArea(contours[i])
    if area > minarea:
        cnt.append(contours[i])
        areas.append(area)

areas.remove(max(areas)) #remove a área da imagem, pois ela é considerada um grão
print(areas)

#---------------------Histograma---------------------#
x = np.array(areas)
fig, ax = plt.subplots()
step1 = 5
step2 = 25
intervals = np.arange(0, 450+step1, step1)
n, bins, patches = ax.hist(x, intervals, rwidth=0.75)
# ax.set_xlabel(r'Área dos Grãos $(\mu m^2)$')
ax.set_xlabel(r'Área dos Grãos $(pixel^2)$')
ax.set_ylabel('Nº de Grãos')
ax.set_title('Histograma de Frequência')
ax.set_xticks(np.arange(0, 450+2*step2, 2*step2))
ax.set_yticks(np.arange(0, 350+2*step2, 2*step2))
ax.axis([0, 450, 0, 350])
#----------------------------------------------------#

img_f = cv2.drawContours(img.copy(), cnt, -1, (0, 0, 255), 2) #destaca os contornos identificados

#---Altera as dimensões da imagem para melhor visualização---#
height = img.shape[0]
width = img.shape[1]
new_size = 0.8
dim = (int(width*new_size), int(height*new_size))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_f = cv2.resize(img_f, dim, interpolation = cv2.INTER_AREA)
edges = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)
#------------------------------------------------------------#

cv2.imshow("Original image", img)
cv2.imshow("Image with contours drawn", img_f)
cv2.imshow("Image with Canny, Dilate and Erode applied", edges)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
