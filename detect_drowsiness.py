#Usage: detect_drowsiness.py
#Import necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

#YAWN_THRESH is used for yawning threshold value
#YAWN_CONSEC_FRAMES is suggest minimum number of frames in which user continuously yawns
#BLINK_THRESH is used for binking threshold value
#BLINK_CONSEC_FRAMES is suggest minimum number of frames in which user continuously blinks
YAWN_THRESH=23.00
YAWN_CONSEC_FRAMES = 26
BLINK_THRESH=7.55
BLINK_CONSEC_FRAMES = 15

#YCOUNTER, BCOUNTER keep track on continuous yawning and blinking 
YCOUNTER = 0
BCOUNTER = 0
#path of a .wav file
alarm= 'F:/Drowsiness_Detection/Trials/alarm.wav'
ALARM_ON = False

#initialize dlib's face detector (HOG-based) and then initialize the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('F:/Drowsiness_Detection/Trials/shape_predictor_68_face_landmarks.dat')

#grab the landmarks of mouth, above part of top lip, below part of top lip, above part of  bottom lip and below part of bottom lip
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(ATopLipStart, ATopLipEnd) = face_utils.FACIAL_LANDMARKS_IDXS["top_lip_above"]
(BTopLipStart, BTopLipEnd) = face_utils.FACIAL_LANDMARKS_IDXS["top_lip_below"]
(ABottomLipStart, ABottomLipEnd) = face_utils.FACIAL_LANDMARKS_IDXS["bottom_lip_above"]
(BBottomLipStart, BBottomLipEnd) = face_utils.FACIAL_LANDMARKS_IDXS["bottom_lip_below"]

#grab the landmark of right eye, left eye, above part of right eye, below part of right eye, above part of left eye, below part of right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(ARightEyeStart, ARightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye_above"]
(BRightEyeStart, BRightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye_below"]
(ALeftEyeStart, ALeftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye_above"]
(BLeftEyeStart, BLeftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye_below"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize it and convert it into gray scale
	frame = vs.read()
	frame = imutils.resize(frame, width=650)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region
		# convert the facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		#draw a rectangle where face region is detected
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
		# extract the mouth, above part of top lip, below part of top lip, above part of  bottom lip and below part of bottom lip coordinates
		mouth = shape[mStart:mEnd]
		TopLipAbove = shape[ATopLipStart:ATopLipEnd]
		TopLipBelow = shape[BTopLipStart:BTopLipEnd]
		BottomLipAbove = shape[ABottomLipStart:ABottomLipEnd]
		BottomLipBelow = shape[BBottomLipStart:BBottomLipEnd]
		TopLip=TopLipAbove
		BottomLip=BottomLipAbove

		#concate coordinates of top lip
		np.concatenate((TopLip, TopLipBelow), axis=None)
		#concate coordinates of bottom lip
		np.concatenate((BottomLip, BottomLipBelow), axis=None)
		#calculate mean of top lip coordinates
		TopLipMean = np.mean(TopLip, axis=0)
		#calculate mean of bottom lip coordinates
		BottomLipMean = np.mean(BottomLip, axis=0)
		#calculate distance between upper lip and lower lip for yawning detection
		YAWN = round(dist.euclidean(TopLipMean, BottomLipMean),2)
		

		#extract the right eye, left eye, above part of right eye, below part of right eye, above part of left eye, below part of right eye coordinates
		LeftEye = shape[lStart:lEnd]
		RightEye = shape[rStart:rEnd]		
		RightEyeAbove = shape[ARightEyeStart:ARightEyeEnd]
		RightEyeBelow = shape[BRightEyeStart:BRightEyeEnd]
		LeftEyeAbove = shape[ALeftEyeStart:ALeftEyeEnd]
		LeftEyeBelow = shape[BLeftEyeStart:BLeftEyeEnd]

                #calculate the mean of above part of right eye, below part of right eye, above part of left eye, below part of right eye coordinates 
		ARightEyeMean = np.mean(RightEyeAbove, axis=0)
		BRightEyeMean = np.mean(RightEyeBelow, axis=0)
		ALeftEyeMean = np.mean(LeftEyeAbove, axis=0)
		BLeftEyeMean = np.mean(LeftEyeBelow, axis=0)

		#calculate distance of above part of right eye and below part of right eye
		LeftEyeBlink = round(dist.euclidean(ARightEyeMean, BRightEyeMean),2)
		#calculate distance of above part of left eye and below part of left eye
		RightEyeBlink = round(dist.euclidean(ALeftEyeMean, BLeftEyeMean),2)
		
		#hull for mouth, left eye and right eye
		mouthHull = cv2.convexHull(mouth)
		LeftEyeHull = cv2.convexHull(LeftEye)
		RightEyeHull = cv2.convexHull(RightEye)

		#draw a contours for mouth, left eye and right eye
		cv2.drawContours(frame, [mouthHull], -1, (255, 255, 0), 1)
		cv2.drawContours(frame, [LeftEyeHull], -1, (255, 255, 0), 1)
		cv2.drawContours(frame, [RightEyeHull], -1, (255, 255, 0), 1)

		#draw a circle for TopLipMean, BottomLipMean,ARightEyeMean, BRightEyeMean, ALeftEyeMean, BLeftEyeMean
		cv2.circle(frame,(int(TopLipMean[0]),int(TopLipMean[1])), 3, (255, 255, 0), -1)
		cv2.circle(frame,(int(BottomLipMean[0]),int(BottomLipMean[1])), 3, (255, 255, 0), -1)
		cv2.circle(frame,(int(ARightEyeMean[0]),int(ARightEyeMean[1])), 2, (255, 255, 0), -1)
		cv2.circle(frame,(int(BRightEyeMean[0]),int(BRightEyeMean[1])), 2, (255, 255, 0), -1)
		cv2.circle(frame,(int(ALeftEyeMean[0]),int(ALeftEyeMean[1])), 2, (255, 255, 0), -1)
		cv2.circle(frame,(int(BLeftEyeMean[0]),int(BLeftEyeMean[1])), 2, (255, 255, 0), -1)

		#yawning contion: distance between lips go above threshold
		YAWNING = YAWN > YAWN_THRESH
		#blinking condition: distance between above part and below part of eyes go below threshold
		BLINKING = LeftEyeBlink < BLINK_THRESH and RightEyeBlink < BLINK_THRESH


		# check for yawning or blinking or for both
		if YAWNING  or BLINKING :
                        if YAWNING:
                                #increase yawning counter if distance between lips go above threshold
                                YCOUNTER += 1
                        
                        if BLINKING:
                                #increase yawning counter if distance between above part and below part of eyes go below threshold
                                BCOUNTER += 1
                                
                        #check for number of frames go above BLINK_CONSEC_FRAMES and YAWN_CONSEC_FRAMES, and if condition satisfied then alarm is on
                        if BCOUNTER >= BLINK_CONSEC_FRAMES or YCOUNTER >= YAWN_CONSEC_FRAMES:
                                if not ALARM_ON:
                                        ALARM_ON = True

                                        if alarm != "":
                                                #create a new thread for alert and play a alarm sound
                                                t= Thread(target=sound_alarm, args=(alarm, ))
                                                t.deamon = True
                                                t.start()
                                                
                                #to display Alert Message on a frame
                                cv2.putText(frame, "ALERT!!!",(280, 45),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102 , 102, 255), 2)
                            
		# reset the counter and alarm
		else:
			YCOUNTER = 0
			BCOUNTER = 0
			ALARM_ON = False

		#to display YAWN, RightEyeBlink and LeftEyeBlink on a frame
		cv2.putText(frame, "Yawn: {:.2f}".format(YAWN), (260, 460),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(frame, "Right Eye Blink: {:.2f}".format(RightEyeBlink), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 , 255, 0), 2)
		cv2.putText(frame, "Left Eye Blink: {:.2f}".format(LeftEyeBlink), (400, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 , 255, 0), 2)
		

				

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do cleanup
cv2.destroyAllWindows()
vs.stop()

