import cv2
import time

captura = cv2.VideoCapture('video.mp4')
num_frames = 196

while (captura.isOpened()):

	for i in range(0, num_frames):
		ret, frame = captura.read()

		if ret == True:
			cv2.imshow('video', frame)
			if cv2.waitKey(30) == ord('s'):
				break
		else: break

captura.release()


#for i in range(0, num_frames):
#	ret, frame = captura.read()

cv2.destroyAllWindows()