# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:29:41 2022

@author: Beyza
"""

import cv2
import face_recognition 

pathTest = "images-video/elizabeth-holmes-test.jpg"
image = cv2.imread(pathTest)
color = (0,0,255)
font = cv2.FONT_ITALIC()

pathElizabeth = "images-video/elizabeth-holmes.jpg"
elizabethImage = face_recognition.load_image_file(pathElizabeth)
elizabethImageEncodings = face_recognition.face_encodings(elizabethImage)[0]
#Yukardaki 0 değerinin amacı bulduğu ilk yüzü atasın.

encodingsList = [elizabethImageEncodings]
namesList = ["Elizabeth Holmes"]


testImage = face_recognition.load_image_file(pathTest)
faceLocations = face_recognition.face_locations(testImage)
faceEncodings = face_recognition.face_encodings(testImage,faceLocations)

for faceLoc, faceEncoding in zip(faceLocations,faceEncodings):
    topLeftY,downRightX,downRightY,topLeftX = faceLoc
    matchedFaces = face_recognition.compare_faces(encodingsList,faceEncoding)
    
    if True in matchedFaces:
      matchedIndex = matchedFaces.index(True)
      name = namesList(matchedIndex)
      
    cv2.rectangle(image, (topLeftX,topLeftY), (downRightX,downRightY), color,1)  
    cv2.putText(image, name, (topLeftX,topLeftY), font, 1, color ,1)
    
    cv2.imshow("Face Recognition", image)