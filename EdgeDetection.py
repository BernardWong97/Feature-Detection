import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 3
ncols = 2
KernelSizeWidth = 5
KernelSizeHeight = 5

img = cv2.imread('GMIT.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayScaledGMIT.jpg', gray_img)
# cv2.imshow('coloured', img)
# cv2.imshow('grayed', gray_img)
# cv2.waitKey(0)                 # Waits forever for user to press any key
# cv2.destroyAllWindows()        # Closes displayed windows

# Blur images
img3x3 = cv2.GaussianBlur(gray_img, (KernelSizeWidth, KernelSizeHeight), 0)

KernelSizeWidth = 5
KernelSizeHeight = 5
img5x5 = cv2.GaussianBlur(gray_img, (KernelSizeWidth, KernelSizeHeight), 0)

KernelSizeWidth = 9
KernelSizeHeight = 9
img9x9 = cv2.GaussianBlur(gray_img, (KernelSizeWidth, KernelSizeHeight), 0)

KernelSizeWidth = 13
KernelSizeHeight = 13
img13x13 = cv2.GaussianBlur(gray_img, (KernelSizeWidth, KernelSizeHeight), 0)

# Gradient X and Y
sobelHorizontal = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)  # x dir
sobelVertical = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)  # y dir
sobelImg = cv2.Sobel(gray_img, cv2.CV_64F, 1, 1, ksize=5)

# Canny
canny = cv2.Canny(gray_img, 100, 200)

# Subplot images
plt.subplot(nrows, ncols, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('GrayScale')
plt.xticks([])
plt.yticks([])

# plt.subplot(nrows, ncols, 3)
# plt.imshow(img3x3, cmap='gray')
# plt.title('3x3')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(nrows, ncols, 4)
# plt.imshow(img5x5, cmap='gray')
# plt.title('5x5')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(nrows, ncols, 5)
# plt.imshow(img9x9, cmap='gray')
# plt.title('9x9')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(nrows, ncols, 6)
# plt.imshow(img13x13, cmap='gray')
# plt.title('13x13')
# plt.xticks([])
# plt.yticks([])

plt.subplot(nrows, ncols, 3)
plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel Horizontal')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 4)
plt.imshow(sobelVertical, cmap='gray')
plt.title('Sobel Vertical')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 5)
plt.imshow(sobelImg, cmap='gray')
plt.title('Sobel Sum')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 6)
plt.imshow(canny, cmap='gray')
plt.title('Canny Image')
plt.xticks([])
plt.yticks([])
plt.show()

