import cv2
import numpy as np

#Open image and resize
img = cv2.imread('beta.jpeg')
#img = cv2.imread('fractal.jpg')
#img = cv2.imread('fish.jpg')


width_im = int(img.shape[1]*0.5)
height_im = int(img.shape[0]*0.5)
dsize = (width_im, height_im)
img_rescale = cv2.resize(img, dsize)
h, w = img_rescale.shape[:2]

#Apply filters to the image for an easy manipulation (gray and thresh) and to eliminate noise (gauss and dilero). Canny for borders.
gray_img = cv2.cvtColor(img_rescale, cv2.COLOR_BGR2GRAY)
denoi = cv2.fastNlMeansDenoising(gray_img, 31, 7, 21) #Filtro para quitar ruido
thresh_img = cv2.adaptiveThreshold(denoi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 53, 15)
img_dil = cv2.dilate(thresh_img, None, iterations=1)

#Find the contours in image and then draw them on the actual image.
cnts,_ = cv2.findContours(img_dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img_rescale, cnts, -1, (0,255,0), 2)

#Obtain areas of each object in the image and append it to a list.
areas_lst = []
for c in cnts:
    area_cnts = cv2.contourArea(c)
    areas_lst.append(float(area_cnts))

#Take the two greater values and draw the contours of both of them.
areas_lst.sort()
max_num1 = areas_lst[-1]
max_num2 = areas_lst[-2]
for c in cnts:
    area_cnts = cv2.contourArea(c)
    if area_cnts == max_num1:
        outer = c
        cv2.drawContours(img_rescale, [outer], -1, (0,255,0), 2)
    if area_cnts == max_num2:
        inner = c
        cv2.drawContours(img_rescale, [inner], -1, (255,0,0), 2)

#Draw the contours in a new black and white image and calculate the center of the figure.
blank = np.zeros(img_rescale.shape, dtype='uint8')
cv2.drawContours(blank, [outer], -1, (255, 255, 255), 2)
cv2.drawContours(blank, [inner], -1, (255, 255, 255), 2)
canny_img = cv2.Canny(blank, 255, 255)

M = cv2.moments(canny_img)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
cv2.circle(img_rescale, (cx, cy), 5, (0,0,255), -1)
cv2.putText(img_rescale, "Center", (cx-20, cy-20), cv2.FONT_ITALIC, 0.5, (0,0,0), 1)

#Calculate polar distances and obtain the max and min value inside the blob.
############################################################################################
# create zeros mask 2 pixels larger in each dimension
mask=np.zeros([h + 2,w + 2], np.uint8)


# floodfill white between two polygons at 240,240
#ffimg= thresh.copy()
#ffimg= thresh.copy()
ffimg=cv2.floodFill(canny_img,mask, (0,0), 0)[1]

############################################################################################
#def_img = blank.astype(dtype='uint8')
distimg = cv2.distanceTransform(ffimg, cv2.DIST_L2, 5)
polar = cv2.warpPolar(distimg, dsize, (cx,cy), 500, cv2.INTER_CUBIC + cv2.WARP_POLAR_LINEAR)


polar_max = np.amax(polar, axis = 1)
#
max_p = 2*np.amax(polar_max)
min_p = 2*np.amin(polar_max)
print(f"Max spacing: {max_p}")
print(f"Min spacing: {min_p}\n")

cv2.imshow('Noise', denoi)
#cv2.imshow('PillThresh', thresh_img)
#cv2.imshow('PillDilEr', img_dil)
#cv2.imshow('PillDilEr', img_ero)
cv2.imshow('Canny', canny_img)
cv2.imshow('Blank', blank)
cv2.imshow('Cont', img_rescale)

cv2.imwrite('prueba/beta.jpg',img_rescale )

cv2.waitKey(0)
cv2.destroyAllWindows()

#https://stackoverflow.com/questions/36504073/how-to-find-the-distance-between-two-concentric-contours-for-different-angles
#https://stackoverflow.com/questions/57056914/how-to-find-distance-between-two-contours
