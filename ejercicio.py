import cv2

img =  cv2.imread("icecream.jpg")

gray =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

scale = 1.5
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resizedgray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

cv2.imshow("imagen",img)
cv2.imshow("gris",gray)
cv2.imshow("Resized image", resized)
cv2.imshow("Resized gray image",resizedgray)

cv2.waitKey(0)
cv2.destroyAllWindows()