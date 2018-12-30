import cv2 
import numpy as np

# Haar Cascase - deep learning 
# Face - Yes, No , 25 stages ke baad detect kar leta hai


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
    
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #scaling factor , min no of frame 
    if len(faces) == 0:
        continue
    
    for face in faces[:1]:
        x,y,w,h = face
        offset = 10
        face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection = cv2.resize(face_offset,(100,100))
        
        cv2.imshow("Face", face_selection)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow("faces",frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    
# gray scale is easy to detect
#min no of neighbours 