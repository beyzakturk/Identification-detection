# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:51:45 2022

@author: Beyza
"""

import cv2
import imutils 
import face_recognition 

pathTest = "images-video/elon_musk.mp4"
image = cv2.imread(pathTest)
color = (0,0,255)
font = cv2.FONT_ITALIC()


cap = cv2.VideoCapture(pathTest)
pathElizabeth = "images-video/elizabeth-holmes.jpg"
elizabethImage = face_recognition.load_image_file(pathElizabeth)
elizabethImageEncodings = face_recognition.face_encodings(elizabethImage)[0]
#Yukardaki 0 değerinin amacı bulduğu ilk yüzü atasın.

encodingsList = [elizabethImageEncodings]
namesList = ["Elizabeth Holmes"]


while True:
    
    ret,frame = cap.read()
    if ret == False:
        break
    #matrisi ne kadar küçültürsek algoritma o kadar hızlanır.Bu yüzden küçültme işlemi yapalım
    row, column , channel = frame.shape
    coefficient =4
    currentColumn = int(column/coefficient)
    frame = imutils.resize(frame,width=currentColumn)

    
    faceLocations = face_recognition.face_locations(frame)
    faceEncodings = face_recognition.face_encodings(frame,faceLocations)
    
    for faceLoc, faceEncoding in zip(faceLocations,faceEncodings):
        topLeftY,downRightX,downRightY,topLeftX = faceLoc
        matchedFaces = face_recognition.compare_faces(encodingsList,faceEncoding)
        
        if True in matchedFaces:
          matchedIndex = matchedFaces.index(True)
          name = namesList(matchedIndex)
          
        cv2.rectangle(frame, (topLeftX,topLeftY), (downRightX,downRightY), color,1)  
        cv2.putText(frame, name, (topLeftX,topLeftY), font, 1/(coefficient/1.5), color ,1)
        
        cv2.imshow("Face Recognition", image)
        
        if cv2.waitKey(1) &  0xFF == ord("c"):
            break
        
cap.release()
cv2.destroyAllWindows()
        
        
        