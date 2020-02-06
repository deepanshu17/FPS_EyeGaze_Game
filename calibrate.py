import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"));
my_head_pose_estimator.load_pitch_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"));
my_head_pose_estimator.load_yaw_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"));

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

filename = "user1.txt"
#print filename
f = open(filename,'w')
f.write("Frame Roll Pitch Yaw\n")

img_folder_path = "user/"
dirListing = os.listdir(img_folder_path)
l = len(dirListing)-1;
#print(len(dirListing));
i = 0;
passed_frames = 0

while(i < l):
	passed_frames += 1
	image = "user/image"+str(i+1)+".jpg"
	frame = cv2.imread(image);
	frame = imutils.resize(frame, width = 450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# print(gray.shape)
	# detect faces in the grayscale frame
	out_face = np.zeros_like(frame)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# print(rects)
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		remapped_shape = np.zeros_like(shape)
		#print(i, remapped_shape.shape)
		feature_mask = np.zeros((frame.shape[0], frame.shape[1]))

		remapped_shape = cv2.convexHull(shape)
		#cv2.drawContours(frame, [remapped_shape], -1, (0, 255, 0), 1)
		#im=frame[remapped_image]

		cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
		feature_mask = feature_mask.astype(np.bool)
		out_face[feature_mask] = frame[feature_mask]
		crop_img = out_face[y:y+h,x:x+w]
	#cv2.imshow("mask_inv", crop_img)
	#cv2.imwrite("out_face.png", out_face)
	#cv2.imshow("Frame", frame)
	if(np.sum(crop_img) == 0):
		print('face not detected')
		continue;
	else:
		image = crop_img
		image = cv2.resize(image,(200,200))
		# Get the angles for roll, pitch and yaw
		roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
		pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
		yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
		print("Estimated [roll, pitch, yaw] ..... [" +str(i+1)+","+ str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
		f.write(str(i) + " " + str(roll[0,0,0]) + " " + str(pitch[0,0,0]) + " " + str(yaw[0,0,0]) +"\n")
		i = i + 1
	
		img_msg = np.zeros((900,1600,3), np.uint8)
		font = cv2.FONT_HERSHEY_SIMPLEX
		message = "Calibrating Pose"
		process = "Processing: " + str(int(passed_frames/9)) + "%"
		cv2.putText(img_msg, message,(200, 300), font, 5,(255,255,255), 8,cv2.LINE_AA)
		cv2.putText(img_msg, process,(450, 600), font, 3,(255,255,255), 6,cv2.LINE_AA)
		cv2.imshow('image', img_msg)

	# cv2.imshow('image', img_msg)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()

f.close()
