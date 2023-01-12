import numpy as np
import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Negative
image = cv2.imread("images/cuprija.jpg", 0)/255
print(image)

negative = 1 - image

fig, ax = plt.subplots(1, 2, figsize = [7, 3.5])
ax[0].imshow(image, cmap = 'gray', aspect = 'auto')
ax[0].axis("off")
ax[0].set_title("Izvorna slika")
cnt, bins, _ = ax[1].hist(image.flatten(), bins = 50, density=True)
ax[1].grid()
ax[1].set_title('Histogram izvorne slike')

fig, ax = plt.subplots(1, 2, figsize = [7, 3.5])
ax[0].imshow(negative, cmap = 'gray', aspect = 'auto')
ax[0].axis("off")
ax[0].set_title("Negativ")
cnt, bins, _ = ax[1].hist(negative.flatten(), bins = 50, density = True)
ax[1].grid()
ax[1].set_title("Histogram negativa")

# Brightness
imageB = np.sqrt(image)

fig, ax = plt.subplots(1, 2, figsize = [7, 3.5])
ax[0].imshow(imageB, cmap = 'gray', aspect = 'auto')
ax[0].axis("off")
ax[0].set_title("Korijen")
cnt, bins, _ = ax[1].hist(imageB.flatten(), bins = 50, density = True)
ax[1].grid()
ax[1].set_title("Histogram korjenovane slike")

plt.show()

# Histogram equalization
image = cv2.imread("images/forestbw.jpg", 0)

plt.figure(figsize=[10,10])
plt.imshow(image, cmap='gray')
plt.axis('off')

fig, ax = plt.subplots(1, 2, figsize = [8, 4.5])
ax[0].imshow(image, cmap = 'gray', aspect = 'auto')
ax[0].axis("off")
ax[0].set_title("Izvorna slika")
cnt, bins, _ = ax[1].hist(image.flatten(), bins = 50, density=True)
ax[1].grid()
ax[1].set_title('Histogram izvorne slike')

imageEq = cv2.equalizeHist(image)

fig, ax = plt.subplots(1, 2, figsize = [8, 4.5])
ax[0].imshow(imageEq, cmap = 'gray', aspect = 'auto')
ax[0].axis("off")
ax[0].set_title("Transformirana slika")
cnt, bins, _ = ax[1].hist(imageEq.flatten(), bins = 50, density = True)
ax[1].grid()
ax[1].set_title("Histogram transformirane slike")
plt.show()

# RGB image histogram equalisation
img = cv2.imread('images/doramar.jpg')
img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


cv2.imshow("equalizeHist", np.hstack((img, hist_eq)))
cv2.waitKey(0)