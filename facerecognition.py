import cv2    #opencv library

import numpy   #numpy library for array operations

import os   #os library for directory process

haar_file = 'haarcascade_frontalface_default.xml'    #initialise haar algorithm filename

datasets = 'datasets'   #datasets folder name

(images,labels,names,id) = ([],[],{},0)   #initialise images - images inside akkash folder , labels - sub folder position(starts with 0), names - {Akkash:0}


for(subdirs,dirs,files) in os.walk(datasets):    #walk into datasets(dirs) -> Akkash (subdirs) -> 50 images(files)

    for subdir in dirs:    #loop for subdirs

        names[id] = subdir    #Akkash will store in names[0]

        subjectPath = os.path.join(datasets,subdir)    #get path of datasets/Akkash

        for filename in os.listdir(subjectPath):      #list down all files in that path

            path = subjectPath + '/' + filename     #get path = datasets/Akkash/1.png

            label = id

            images.append(cv2.imread(path,0))    #read all files inside datasets/Akkash and store in images array

            labels.append(int(label))     #store label in labels array

        id = id + 1    #id increment

(images,labels) = [numpy.array(lis) for lis in [images,labels]] #store images and labels in array

model = cv2.face.FisherFaceRecognizer_create()   #load fisher algorithm

model.train(images,labels)    #train with images and labels

alg = cv2.CascadeClassifier(haar_file)   #load haar algorithm

camera = cv2.VideoCapture(0)   #initialise primary camera

cnt = 0

while True :     #infinite loop to run camera continuously

    _,img = camera.read()    #read frame from camera

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #convert grayscale img

    faces = alg.detectMultiScale(grayImg , 1.3,4)   #to detect face coordinates

    for(x,y,w,h) in faces:

        face = grayImg[y:y+h, x:x+w ]      #crop face only

        face_resize = cv2.resize(face,(130,100))    #resize img

        prediction = model.predict(face_resize)    #predict o/p of face - prediction declare 2 outputs (0 and 1) 1=confidentlevel 0=id

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   #draw rectangle


        if prediction[1] < 800:   #confident level less than 800

            cv2.putText(img,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,51,255))  #display name with confident level

            cnt =0

        else:

            cnt = cnt +1

            cv2.putText(img,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,51,255))  #display name as unknown if camera detects person not in datasets

            if(cnt > 100):

                cv2.imwrite('input.jpg',img)   #store unknown person img in file

                cnt =0

    cv2.imshow('face',img)   #display image

    key = cv2.waitKey(10)  #wait for 10 frames

    if key == 27:   #when esc key is pressed, ends camera
        break


camera.release()    #release camera

cv2.destroyAllWindows()   #close windows

            
            

    


        

    
                                                             
