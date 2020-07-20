#trainer
#face recognizer
import cv2
import numpy as np
from PIL import Image
import os
from numpy import array
name=['krish','dhruvin']
import matplotlib.pyplot as plt

# Path for face image database
path = '/content/drive/My Drive/lab7ai/actual_image_dataset1'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


# function to get the images and label data
def convertToRGB(image):
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    
    
    for imagePath in imagePaths:

            id = ((imagePath).split(".")[1])
            if(id in name):
              
              if(name[0] in id):
                id=0
              else:
                id=1
              
            else:
              continue
            test_image = cv2.imread(imagePath)
            test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            
            plt.imshow(test_image_gray, cmap='gray')
            haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces_rects = haar_cascade_face.detectMultiScale(test_image_gray,scaleFactor = 1.3, minNeighbors = 2)
            k=0
            
            for (x,y,w,h) in faces_rects:
               
                cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 10)
                faceSamples.append(test_image_gray[y:y+h,x:x+w])
                ids.append(id)
               
           
    return faceSamples,ids

faces,ids = getImagesAndLabels(path)

  
recognizer.train(faces, np.array(ids))

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

#using face extraction from images
path='/content/drive/My Drive/lab7ai/test_own_dataset/test_original'
imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
ans_data=[]
for imagepath in imagePaths:
  id = ((imagepath).split(".")[1])
  test_image = cv2.imread(imagepath)
  test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
  plt.imshow(test_image_gray, cmap='gray')
  haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.3, minNeighbors = 1)

  k=0
  for (x,y,w,h) in faces_rects:
    k=test_image_gray[y:y+h,x:x+w]
  ans=[recognizer.predict(k)]
  val=ans[0][0]
  actual_data=[id]
  ans_data.append([name[val],actual_data])
  plt.imshow(k, cmap='gray')
  print(name[val])
print(ans_data)
wrong=0
for i in ans_data:
  if(i[0]!=i[1][0]):
    wrong=wrong+1
print(1-wrong/len(ans_data))

