import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2

### ====================================== ###
import pygame
from pygame.locals import *
import time
from random import *
from math import *
from statistics import *

################# function callibration ##################################################
def callibrate(src_img_points,dest_img_points):
	#print('src_img_points',src_img_points);
	B=np.linalg.pinv(src_img_points);
	u=np.dot(B,dest_img_points);
	#print(u.shape);
	return u;

############################3  Function to find box around face #########################
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

#################### Create source img points #####################################################3
#src_img_points=np.ones((9,5))
def src_points(c):
	src_img_points=np.ones((l,6))

	count=0;i=0;
	filename1="user" +str(c)+".txt"
	f1=open(filename1);
	lines1=f1.readline()
	lines1=f1.readline()
	x1_read=lines1;i=0;

	filename2="user"+ str(c)+"_pupil.txt"
	f2=open(filename2);
	lines2=f2.readline()
	lines2=f2.readline()
	x2_read=lines2;i=0;
	#print(x)
	for i in range(l):
		lines=lines1.split(" ")
		# print(i, lines)
		src_img_points[i,0]=lines[2]
		src_img_points[i,1]=lines[3]
		lines1=f1.readline()
	
	for i in range(l):
		lines=lines2.split(" ")
		#print(i, lines)
		src_img_points[i,2]=lines[1]
		src_img_points[i,3]=lines[2]
		src_img_points[i,4]=lines[3]
		src_img_points[i,5]=lines[4]
		lines2=f2.readline()
	return src_img_points;

# 2 - Initialize the game
pygame.init()

# MUSIC
# pygame.mixer.music.load("resources/music/Hero_Theme.mp3")
# pygame.mixer.music.play(-1)

# W,H = 1280,720
# W, H = 1580,885
W, H = 1600, 900
screen=pygame.display.set_mode((W, H))
keys = [False, False]
shoot = False
origin=[0,0]

gunpos_x = W/3
gunpos_y = H/1.35
gunpos=[gunpos_x, gunpos_y]

low = floor(W/25)
high = floor(H/2)
step = floor((high-low)/2)
epos_x = randrange(low, high, step)
# epos_x = randrange(w/25, w/2, 23*w/100)
epos_y = H/1.7
epos = [epos_x, epos_y]

zone = pygame.image.load("resources/front.png")
zone = pygame.transform.scale(zone,(W, H))

gun = pygame.image.load("resources/gun.png")
gunsize = gun.get_rect().size
gun_w = floor(gunsize[0]/3)
gun_h = floor(gunsize[1]/3)
gun = pygame.transform.scale(gun, (gun_w,gun_h))

bullet = pygame.image.load("resources/bullet3.png")
bsize = bullet.get_rect().size
b_w = floor(bsize[0]/7)
b_h = floor(bsize[1]/7)
bullet = pygame.transform.scale(bullet, (b_w,b_h))

aim = pygame.image.load("resources/gunpoint.png")
asize = aim.get_rect().size
a_w = floor(asize[0]/10)
a_h = floor(asize[1]/10)
aim = pygame.transform.scale(aim, (a_w,a_h))

enemy = pygame.image.load("resources/enemy.png")
esize = enemy.get_rect().size
e_w = floor(esize[0]/7)
e_h = floor(esize[1]/7)
enemy = pygame.transform.scale(enemy, (e_w, e_h))
eFlag = 1
T = 1

blast = pygame.image.load("resources/blast1.png")
blsize = blast.get_rect().size
bl_w = floor(blsize[0]/3)
bl_h = floor(blsize[1]/3)
blast = pygame.transform.scale(blast, (bl_w,bl_h))
blFlag = 0

LEFT = 1
score = 0

############################## Headpose detection ###########################################
sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"))


##############################   compute thw no of images collected ################################################################
img_folder_path = "user/"
dirListing = os.listdir(img_folder_path)
l = len(dirListing) - 1; 
i = 0;
#print(len(dirListing));

##############################    print("[INFO] loading facial landmark predictor...") #############################################
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lol.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#print(rStart, rEnd)


################# Create dest img points ########################################################################################
im2=np.zeros((H, W, 3))
im2_s=im2.shape
cell_w=im2_s[1]/3
cell_h=im2_s[0]/3
pt1=cell_w/2; pt2=cell_h/2;
k=0;i=0;j=0;

dest_points=np.ones((9,3))
for i in range(3):
	for j in range(3):
		# print (k);
		dest_points[k,0]=pt1+j*cell_w; dest_points[k,1]=pt2+i*cell_h;
		k=k+1

points=np.zeros((l,2));
dest_img_points=np.ones((l,3))
# print(dest_points)

for i in range(100):
	points[i,0]=dest_points[0,0]; points[i,1]=dest_points[0,1];
	points[i+100,0] = dest_points[1,0]; points[i+100,1] = dest_points[1,1];
	points[i+200,0] = dest_points[2,0]; points[i+200,1] = dest_points[2,1];
	points[i+300,0] = dest_points[3,0]; points[i+300,1] = dest_points[3,1];
	points[i+400,0] = dest_points[4,0]; points[i+400,1] = dest_points[4,1];
	points[i+500,0] = dest_points[5,0]; points[i+500,1] = dest_points[5,1];
	points[i+600,0] = dest_points[6,0]; points[i+600,1] = dest_points[6,1];
	points[i+700,0] = dest_points[7,0]; points[i+700,1] = dest_points[7,1];
	points[i+800,0] = dest_points[8,0]; points[i+800,1] = dest_points[8,1];


dest_img_points[:,:-1]=points
dest = dest_img_points
#dest=np.concatenate((dest_img_points, dest_img_points, dest_img_points, dest_img_points, dest_img_points),axis=0)
#print(dest.shape)
#print("dest_img_points")
#print(dest_img_points);


#print("dest_img_points")
#print(dest_img_points.shape);

c=0
src=[]
for c in range(1):
	# print(c+1)
	#src=np.concatenate((src,src_points(c+1)),axis=0)
	src_temp=src_points(c+1)
	if(c==0):
		src=src_temp
	else:
		src=np.concatenate((src, src_temp),axis=0)
	# print (src.shape)

src_img_points=src
dest_img_points=dest

############### callibrate ##########################################################3
u=callibrate(src,dest)
#print(u.shape)
pred_points=np.dot(src[:,:],u)
#print(pred_points)

#count=0


################### plot the points on image ############################################
for i in range(l):
	cv2.circle(im2,(int(dest[i,0]), int(dest[i,1])),20,(0,0,255),-1)
	cv2.circle(im2,(int(pred_points[i,0]), int(pred_points[i,1])),15,(0,255,0),-1)

#print(src_img_points[:,2:-1],pred_points[:,:-1])

cv2.imshow('image',im2);
cv2.imwrite('cali_image.png',im2);
cv2.waitKey(2000)
cv2.destroyAllWindows()

#################################################### Run live video and keep on updating callibration matrix ################################3
# start the video stream thread
#rint("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
fileStream = True
vs =VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

pred_points_sum=np.zeros((1,2))
#final_pred_points=[im2.shape[0],im2.shape[1]]
final_pred_points=[]

sx = []
sy = []
tx, ty = 0, 0

start_time = time.time()
clock = pygame.time.Clock()

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	
	screen.fill(0)
	
	cx = pred_points[0,0]
	cy = pred_points[0,1]
	if len(sx) < 5:
		sx.append(cx)	#Sequence of X - Two points at a moment
		sy.append(cy)
	else:
		sx.append(cx)
		sy.append(cy)
		del sx[0]
		del sy[0]

	# print(sx)
	tx = mean(sx)	#Target_X
	ty = mean(sy)
	
	gunpos=[tx, H/1.35]
	apos = [gunpos[0]+W/30, gunpos[1]-W/30]

	screen.blit(zone,origin)
	if eFlag:
	    screen.blit(enemy, epos)

	# show blast    
	if blFlag and T+2:
	    screen.blit(blast, blpos)
	    T = T-1

	# show enemy back
	if not T:
	    eFlag = 1

	screen.blit(aim, apos)
	screen.blit(gun, gunpos)

	# SHOW SCORE
	colour = (200, 50, 0)
	font = pygame.font.Font(None, 50)
	message = "Score: " + str(score)
	text = font.render(message, 1, colour)
	screen.blit(text, (80*W/100, W/50))

	# SHOW TIME
	time_consumed=round((time.time() - start_time), 2)
	time_taken="Time: " + str(time_consumed) + "s"
	text2 = font.render(time_taken , True, colour)
	screen.blit(text2, (80*W/100, W/25))
	pygame.display.flip()

	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			pygame.quit()
			exit(0)

		if event.type == pygame.KEYDOWN:
			if event.key==K_q:
				pygame.quit()
				exit(0)
		
		if event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
			effect = pygame.mixer.Sound('resources/music/gunsound.wav')
			effect.play(0)


	ax = apos[0]
	ay = apos[1]

	if ax>epos[0] and ax<epos[0]+e_w and ay>epos[1] and ay<epos[1]+e_h:
		effect = pygame.mixer.Sound('resources/music/gunsound.wav')
		effect.play(0)
		score += 10
		if score == 150:
			font = cv2.FONT_HERSHEY_SIMPLEX
			img = np.zeros((900,1600,3), np.uint8)
			cv2.putText(img, "You made it!",(300,400), font, 5, (0,255,50), 9, cv2.LINE_AA)
			cv2.putText(img, time_taken, (500,600), font, 3, (255,255,255), 6, cv2.LINE_AA)
			cv2.imshow('GAME OVER', img)
			cv2.waitKey(3000)

			pygame.quit()
			exit(0)

		blFlag = 1
		blpos = epos
		T = 5
		low = floor(W/25)
		high = floor(W/2)
		step = floor((high-low)/2)
		# epos_x = randint(low, high)
		epos_x = randrange(low, high, step)
		epos_y = H/1.7
		epos = [epos_x, epos_y]
		eFlag = 0

	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	out_face=np.zeros_like(frame)
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

		################################################  Pupil location ####################################################
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

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		remapped_shape=np.zeros_like(shape)
		#print(remapped_shape.shape)
		feature_mask=np.zeros((frame.shape[0], frame.shape[1]))

		remapped_shape=cv2.convexHull(shape)
		#cv2.drawContours(frame, [remapped_shape], -1, (0, 255, 0), 1)
		#im=frame[remapped_image]

		cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
		feature_mask = feature_mask.astype(np.bool)
		out_face[feature_mask] = frame[feature_mask]
		crop_img=out_face[y:y+h,x:x+w]
   
	#cv2.imshow("mask_inv", crop_img)
	#cv2.imwrite("out_face.png", out_face)
	#cv2.imshow("Frame", frame)

	image=crop_img
	image = cv2.resize(image,(200,200))
    # Get the angles for roll, pitch and yaw
	roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
	pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
	yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
	#print("Estimated [roll, pitch, yaw] ..... [" +str(i)+","+ str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")

	#im2=np.zeros((1000,1800,3))
	pred_points=[]
	temp_points=np.ones((1,6))
	#lines=lines1.split(" ")
	temp_points[0,0]=pitch
	temp_points[0,1]=yaw
	temp_points[0,2]=pt_x
	temp_points[0,3]=pt_y
	temp_points[0,4]=ptr_x
	temp_points[0,5]=ptr_y
	#print(temp_points[:,2:])
	u=callibrate(src_img_points,dest_img_points)
	pred_points=np.dot(temp_points,u)
	pred_points_sum[0,0] = pred_points_sum[0,0] + pred_points[0,0]
	pred_points_sum[0,1] = pred_points_sum[0,1] + pred_points[0,1]
	#print("pred_points",pred_points)
	src_img_points=np.append(src_img_points,temp_points,axis=0)
	dest_img_points=np.append(dest_img_points,pred_points,axis=0)
	#print(src_img_points.shape)
	# n=2

	# if(i % n == 0):
	# 	##3################## average value of n frames ###############################################3
	# 	final_pred_points = pred_points_sum/n
	# 	# print(final_pred_points.shape)
	# 	print(final_pred_points[0,0],final_pred_points[0,1])
	# 	#########################3   store avg of two frames in shoot #######################
	# 	shoot[shoot_count2,0]=final_pred_points[0,0]; shoot[shoot_count2,1] = final_pred_points[0,1];
	# 	shoot_count2+=1
	# 	im2=np.zeros((1000,1800,3))
	# 	cv2.circle(im2,(int(final_pred_points[0,0]), int(final_pred_points[0,1])),45,(0,255,0),-1)
	# 	#i=i+1
	# 	#screen.blit(aim,(int(final_pred_points[0,0]), int(final_pred_points[0,1])))
	# 	pygame.draw.circle(screen, (0,255,0),(int(final_pred_points[0,0]), int(final_pred_points[0,1])), 40 ,0)
	# 	pred_points_sum = np.zeros((1,2));

	# 	#lines1=f1.readline()
	# 	#x=lines1;
	# 	i=i+1;
	# else:
	# 	#im2=np.zeros((1000,1800,3))
	# 	cv2.circle(im2,(int(final_pred_points[0,0]), int(final_pred_points[0,1])),45,(0,255,0),-1)
	# 	#screen.blit(aim,(int(final_pred_points[0,0]), int(final_pred_points[0,1])))
	# 	pygame.draw.circle(screen, (0,255,0),(int(final_pred_points[0,0]), int(final_pred_points[0,1])), 40 ,0)
	# 	#lines1=f1.readline()
	# 	#x=lines1;
	# 	i=i+1

	#print(src_img_points.shape)

	# cv2.imshow('image',im2);

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
