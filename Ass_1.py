import cv2 as cv
import numpy as np




def rescaleFrame(frame, scale=0.75):
    width =int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# 1
img = cv.imread('C:/Users/rajan/Desktop/Images/1.jpg')
img = rescaleFrame(img, scale=0.50)
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)


canny1 = cv.Canny(img, 125,175)
canny2 = cv.Canny(blur, 125,175)
cv.imshow('house', img)
cv.imshow('house_Edge', canny1)
cv.imshow('house_blur', canny2)
cv.waitKey(0)

# 2
img_2 = cv.imread('C:/Users/rajan/Desktop/Images/6.jpg')
cv.imshow('Holls', img_2)
eroded = cv.erode(img_2, (300,300), iterations=50)

dilated = cv.dilate(eroded,(200,200),iterations=30)
# cv.imshow('Eroded', eroded)
# cv.waitKey(0)

cv.imshow('Dilated', dilated)
cv.waitKey(0)

# 3
img_3 = cv.imread('C:/Users/rajan/Desktop/Images/2.jpg')
gray = cv.cvtColor(img_3, cv.COLOR_BGR2GRAY)

cv.imshow('Angry_man', img_3)

ret, thresh = cv.threshold(gray, 115,255, cv.THRESH_BINARY)

cv.imshow('Bin_img', thresh)
cv.waitKey(0)


# 4
img_4 = cv.imread('C:/Users/rajan/Desktop/Images/15.jpg')
img_4 = rescaleFrame(img_4, scale=0.20)
cv.imshow('Marian_Drive', img_4)

# Gaussian Blur
gauss = cv.GaussianBlur(img_4, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)


# Bilateral Filter
bilateral = cv.bilateralFilter(img_4, 20, 35, 35)
cv.imshow('Bilateral', bilateral)
cv.waitKey(0)


# 5
img_5 = cv.imread('C:/Users/rajan/Desktop/Images/10.jpg')
img_5 = rescaleFrame(img_5, scale=0.75)
cv.imshow('RGB_img', img_5)

b,g,r = cv.split(img_5)
blank = np.zeros(img_5.shape[:2], dtype='uint8')

bgr_img = cv.merge([r,g,b])
cv.imshow('BGR_img', bgr_img)
cv.waitKey(0)