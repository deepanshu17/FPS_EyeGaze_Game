import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
#print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lol.dat")



filename = "user1_pupil.txt"
# print(filename)
f=open(filename,'w')
f.write("Frame Roll Pitch Yaw\n")

# start the video stream thread
#rint("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
#vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
#fileStream = False
time.sleep(1.0)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# print(rStart, rEnd)



img_folder_path = "user/"
dirListing = os.listdir(img_folder_path)
l=len(dirListing)-1
# print(len(dirListing));
i=0;
passed_frames = 0

# loop over frames from the video stream
while(i<l):
	passed_frames += 1
	image_name="user/image"+str(i+1)+".jpg"
	frame = cv2.imread(image_name)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	out_face=np.zeros_like(frame)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = np.array(shape[lStart:lEnd])
		rightEye =np.array( shape[rStart:rEnd])
		#print(leftEye[0,0],(leftEye).shape)

		pt1_x=(np.sum(leftEye[1,0] + leftEye[2,0]))/2.0
		pt1_y=(np.sum(leftEye[1,1] + leftEye[2,1]))/2.0
		pt2_x=(np.sum(leftEye[5,0] + leftEye[4,0]))/2.0
		pt2_y=(np.sum(leftEye[5,1] + leftEye[4,1]))/2.0
		pt_x=int(np.sum(pt1_x + pt2_x)/2.0)
		pt_y=int(np.sum(pt1_y + pt2_y)/2.0)

		pt1r_x=(np.sum(rightEye[1,0] + rightEye[2,0]))/2.0
		pt1r_y=(np.sum(rightEye[1,1] + rightEye[2,1]))/2.0
		pt2r_x=(np.sum(rightEye[5,0] + rightEye[4,0]))/2.0
		pt2r_y=(np.sum(rightEye[5,1] + rightEye[4,1]))/2.0
		ptr_x=int(np.sum(pt1r_x + pt2r_x)/2.0)
		ptr_y=int(np.sum(pt1r_y + pt2r_y)/2.0)
		cv2.circle(frame,(int(pt_x), int(pt_y)),3,(0,0,255),-1)
		cv2.circle(frame,(int(ptr_x), int(ptr_y)),3,(0,0,255),-1)
		# print(pt_x,pt_y)
	
	f.write(str(i) + " " + str(pt_x) + " " + str(pt_y) + " " + str(ptr_x) + " " + str(ptr_y) +"\n")
	i=i+1
	cv2.imshow("Frame", frame)
	
	img_msg = np.zeros((900, 1600,3), np.uint8)
	font = cv2.FONT_HERSHEY_SIMPLEX
	message = "Calibrating Pupil"
	process = "Processing: " + str(int(passed_frames/9)) + "%"
	cv2.putText(img_msg, message,(200, 300), font, 5, (255,255,255), 8, cv2.LINE_AA)
	cv2.putText(img_msg, process,(450, 600), font, 3, (255,255,255), 6,cv2.LINE_AA)
	cv2.imshow('image', img_msg)

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# print(i)

img_msg = np.zeros((900,1600,3), np.uint8)
message2= "Calibration Done"
cv2.putText(img_msg, message2 ,(180,450), font, 5, (0,255,255), 8, cv2.LINE_AA)
cv2.imshow('image', img_msg)
cv2.waitKey(2000)

cv2.destroyAllWindows()


