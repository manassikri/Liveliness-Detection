from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.blinkdetector import BlinkDetector
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
from scipy.spatial import distance as dist
import time
import cv2
from imutils import face_utils
import dlib
import numpy as np

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
global COUNTER
COUNTER = [0]*64
global TOTAL
TOTAL = [0]*64
blink_check=[0]*64

def calculate_centroid(rect):
    (startX, startY, endX, endY)=rect
    cX=int((startX+endX)/2.0)
    cY=int((startY+endY)/2.0)

    return(cX,cY)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
        
    return ear

def main_loop_blink(frame,objects):

    for ob in objects:
        rect=ob[2].tolist()
        rect=dlib.dlib.rectangle(rect[0],rect[1],rect[2],rect[3])
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        shape=predictor(gray,rect)
        #print("gray",gray)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)
        ear=(leftEAR+rightEAR)/2
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        ear=(leftEAR+rightEAR)/2
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

        if ear < EYE_AR_THRESH:
            COUNTER[ob[0]]+=1
        else:
            if COUNTER[ob[0]]>=EYE_AR_CONSEC_FRAMES:
                TOTAL[ob[0]]+=1
            COUNTER[ob[0]]=0
                    
            if TOTAL[ob[0]]>3:

                blink_check[ob[0]] = 1
            else:
                blink_check[ob[0]] = 0

    return blink_check

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")

ap.add_argument("-l", "--shape-predictor", required=True,
                help="path")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())
ct = CentroidTracker()

predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(H, W) = (None, None)
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)
frameList=[]
global blinks
blinks=[0]*64
   
count=0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []
    for i in range(0, detections.shape[2]):
        if detections[0,0,i,2]>args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
    objects=[]
    obj=ct.update(rects)
    centroid_rect_list=[]
    
    
    for rect in rects:
        centroid=calculate_centroid(rect)
        centroid_rect_list.append([centroid,rect])
    
    
    
    for centroid_rect in centroid_rect_list:
        for(objectID, centroid) in obj.items():
            x1,y1=centroid_rect[0]
            x2,y2=centroid

            if x1 == x2 and y1==y2:
                objects.append([objectID, centroid, centroid_rect[1]])
                break

    
    frameList.append([frame,objects])
    # print("////")
    # print("manas")
    # print("*******")
    blinks=main_loop_blink(frame,objects)
    count=count+1
    if count>=10:
        
        
        #print(blinks)
        #print("\\\\\\\\\\\\\\\\\\\\")
        #blinks =main_loop_blink(frameList)
        count=0
        print("10th frme")


    for object_it in objects:
        #print(object_it[1])
        print(object_it[0])
        #print(object_it[2])
        print("************")
        print(TOTAL[object_it[0]])
        
        
        
        #print(object_it)
        text = "ID {}".format(object_it[0])

        #print(object_it[0])
        blink_text="TOTAL{}".format(TOTAL[object_it[0]])

        (x,y)=object_it[1]
        cv2.putText(frame, text, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print(blinks[object_it[0]])
        cv2.putText(frame,blink_text,(x + 10, y + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #cv2.circle(frame, (x, y, 4, (0, 255, 0), -1)
        
                                    
                               
               
        if blinks[object_it[0]] == 1:
                   
            (startX, startY, endX, endY)=object_it[2]
            cv2.rectangle(frame,(startX, startY),(endX, endY),(0,255,0),2)
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)&0xFF

    if key ==ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()