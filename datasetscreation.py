import cv2    #opencv library

import os    #os library for directory oriented process


haar_cascade = "haarcascade_frontalface_default.xml"   #initialise haar cascade algorithm file

datasets = 'datasets'     #folder name where data need to store

sub_data = 'Vijay'    #sub folder to create inside datasets folder

path = os.path.join(datasets, sub_data)    #to create path - datasets/Akkash

if not os.path.isdir(path):           #check whether path is present or folder is present

    os.mkdir(path)          #create folder or make directory


alg = cv2.CascadeClassifier(haar_cascade)   #loading algorithm

camera = cv2.VideoCapture(0)    #initialise primary camera

count = 1    #initiallise count

while count < 51:     #run upto 50 frames - 50 images for training 

    print(count)    #print count

    _,img = camera.read()  # read frame from camera

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #convert to grayscale pic

    faces = alg.detectMultiScale(grayImg,1.3,4)   #detect face coordinates

    for (x,y,w,h) in faces :

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   #draw rectangle around faces

        face = grayImg[y:y+h,x:x+w]    #to crop face only

        face_resize = cv2.resize(face,(130,100))   #resize img

        cv2.imwrite('%s/%s.png' % (path,count), face_resize)    #save every pic in datasets/Akkash/1.png- save img like 1.png

    count = count +1    #count increment

    cv2.imshow('face',img)   #display image

    key = cv2.waitKey(10)   #wait for 10 frames

    if key == 27:       #when esc key is pressed, ends camera
        break

camera.release()  #camera release

cv2.destroyAllWindows()   #close window


    
    



    

    

    

    


