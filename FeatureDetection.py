import cv2
import numpy as np
from matplotlib import pyplot as plt

originalImg = cv2.imread('GMIT1.jpg')
grayGmit1 = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayScaledGMIT1.jpg', grayGmit1)

nrows = 5
ncols = 2
imgHarris = originalImg.copy()
imgShiTomasi = originalImg.copy()
imgSift = originalImg.copy()

# Harris
dst = cv2.cornerHarris(grayGmit1, 2, 3, 0.04)

threshold = 0.09  #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris, (j, i), 3, (255, 0, 255), -1)

# ShiTomasi
corners = cv2.goodFeaturesToTrack(grayGmit1, 100, 0.01, 10)

for i in corners:
    x, y = i.ravel()
    cv2.circle(imgShiTomasi, (x, y), 3, (0, 255, 255), -1)

# Initiate SIFT detector
sift = cv2.SIFT(50)
kp = sift.detect(grayGmit1, None)
# Draw keypoints
imgSift = cv2.drawKeypoints(imgSift, kp, color=(255, 255, 0), flags=4)

# Subplot images
plt.subplot(nrows, ncols, 1)
plt.imshow(cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 3)
plt.imshow(grayGmit1, cmap='gray')
plt.title('Grayscaled')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 5)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Harris')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 7)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Shi Tomasi')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 9)
plt.imshow(cv2.cvtColor(imgSift, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('SIFT')
plt.xticks([])
plt.yticks([])

# 2nd Image
originalStone = cv2.imread('StoneBuilding.jpg')
grayStone = cv2.cvtColor(originalStone, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayScaledStone.jpg', grayStone)

stoneHarris = originalStone.copy()
stoneShiTomasi = originalStone.copy()
stoneSift = originalStone.copy()

# Harris
dst = cv2.cornerHarris(grayStone, 2, 3, 0.04)

threshold = 0.09  #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(stoneHarris, (j, i), 3, (255, 0, 255), -1)

# ShiTomasi
corners = cv2.goodFeaturesToTrack(grayStone, 100, 0.01, 10)

for i in corners:
    x, y = i.ravel()
    cv2.circle(stoneShiTomasi, (x, y), 3, (0, 255, 255), -1)

# Initiate SIFT detector
sift = cv2.SIFT(50)
kp = sift.detect(grayStone, None)
# Draw keypoints
stoneSift = cv2.drawKeypoints(stoneSift, kp, color=(255, 255, 0), flags=4)

# Subplot images
plt.subplot(nrows, ncols, 2)
plt.imshow(cv2.cvtColor(originalStone, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 4)
plt.imshow(grayStone, cmap='gray')
plt.title('Grayscaled')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 6)
plt.imshow(cv2.cvtColor(stoneHarris, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Harris')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 8)
plt.imshow(cv2.cvtColor(stoneShiTomasi, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Shi Tomasi')
plt.xticks([])
plt.yticks([])

plt.subplot(nrows, ncols, 10)
plt.imshow(cv2.cvtColor(stoneSift, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('SIFT')
plt.xticks([])
plt.yticks([])
plt.show()


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out