#### Code to collect data for callibration
import cv2
import numpy as np

W, H = 1600, 900
cap = cv2.VideoCapture(0)
im = np.zeros((H, W, 3))
s = im.shape
i, j = 0, 0
k = 1
count = 0
count2 = 0

cell_w = s[1]/3
cell_h = s[0]/3

pt1 = cell_w/2; pt2 = cell_h/2;

while(i < 3):
	while(j < 3):
		#print(count2)
		if(count2 < 30):
			count2 = count2+1
			ret, frame = cap.read()
			im = np.zeros((H, W, 3))
			cv2.imshow('image', im)
		else:
			#print(i,j)
			ret, frame = cap.read()
			count = count + 1;
			im = np.zeros((H, W, 3))
			cv2.circle(im, (int(pt1 + j*cell_w), int(pt2 + i*cell_h)), 15, (0, 0, 255), -1)
			cv2.imshow('image',im)

			if(count == 100):
				count = 0
				j += 1

			img_name = "user/image" + str(k) + ".jpg"
			# print(img_name)
			cv2.imwrite(img_name, frame)
			count2 += 1
			k += 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# print(i)
	i += 1
	j = 0;

cap.release()
cv2.destroyAllWindows()
