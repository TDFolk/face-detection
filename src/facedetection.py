import sys
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    'C:/Users/Trent/Anaconda3/Library/etc/haarcascades_github/haarcascade_frontalface_alt2.xml'
)
#eye_cascade = cv2.CascadeClassifier('C:/Users/Trent/Anaconda3/Library/etc/haarcascades/haarcascade_eye.xml')
img = cv2.imread(sys.argv[1])
original_img = img.copy()
print(img.size)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

#histStretch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#gray = histStretch.apply(gray)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=1,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

num = 0
for (x,y,w,h) in faces:
    x -= 30
    y -= 40
    w += 70
    h += 70
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    crop = original_img[y:y+h, x:x+w]
    cv2.imwrite('jaycrop5_' + str(num) + '.jpg', crop)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    num += 1

cv2.imwrite('bigresult6.jpg', img)
cv2.waitKey(0)




