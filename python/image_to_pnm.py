import cv2 as cv

img = cv.imread("bike-sample.jpg")

cv.imwrite("bike_sample.pnm", img)
