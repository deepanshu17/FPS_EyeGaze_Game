import cv2
import numpy as np
import pygame
from pygame.locals import *

W, H = 1600, 900
pygame.init()

pygame.mixer.music.load("resources/music/Hero_Theme.mp3")
pygame.mixer.music.play(-1)

img = np.zeros((H, W, 3), np.uint8)
img = cv2.imread('resources/front.png')
img = cv2.resize(img, (W, H))

# FIRST WINDOW - Game Begin Screen
# font = cv2.FONT_HERSHEY_SIMPLEX
font = cv2.FONT_HERSHEY_DUPLEX
Title = "EYE GAZE GAMING"
message = "Press Enter to calibrate"
cv2.putText(img, Title,(200,200), font, 4,(255,0,0), 9, cv2.LINE_AA)
cv2.putText(img, message,(400,600), font, 2,(255,255,255), 4, cv2.LINE_AA)
cv2.imshow('image', img)
# cv2.waitKey(0)

key = cv2.waitKey(0)
# print(key)
# If pressed Enter
if key==13:
	# SECOND WINDOW IN DATA - Calibration points
	import data

	# THIRD WINDOW IN CALIBRATE - Calibrating Pose
	import calibrate

	# FOURTH WINDOW IN PUPIL - Calibrating Pupil
	# FIFTH WINDOW IN PUPIL - Calibration Done
	import pupil

	# SIXTH WINDOW - Waiting For User to Start
	img = np.zeros((900,1600,3), np.uint8)
	cv2.putText(img, "Press Enter to play", (320, 470), font, 3,(0,0,255), 6, cv2.LINE_AA)
	cv2.imshow('image', img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

	# SEVENTH WINDOW - Game Starts
	import eye_gaze
