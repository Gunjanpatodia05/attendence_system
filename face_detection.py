# importing librarys
import cv2
import numpy as np
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# img declaration
gunjan=face_rec.load_image_file('sampe_images/gunjan.jpg')
gunjan = cv2.cvtColor(gunjan, cv2.COLOR_BGR2RGB)
gunjan = resize(gunjan, 0.50)
gunjan_test=face_rec.load_image_file('sampe_images/gunjan_test.jpg')
gunjan_test = cv2.cvtColor(gunjan_test, cv2.COLOR_BGR2RGB)
gunjan_test = resize(gunjan_test, 0.50)

# finding face location

faceLocation_gunjan = face_rec.face_locations(gunjan)[0]
encode_gunjan = face_rec.face_encodings(gunjan)[0]
cv2.rectangle(gunjan, (faceLocation_gunjan[3], faceLocation_gunjan[0]), (faceLocation_gunjan[1], faceLocation_gunjan[2]), (255, 0, 255), 3)

faceLocation_gunjantest = face_rec.face_locations(gunjan_test)[0]
encode_gunjantest = face_rec.face_encodings(gunjan_test)[0]
cv2.rectangle(gunjan_test, (faceLocation_gunjan[3], faceLocation_gunjan[0]), (faceLocation_gunjan[1], faceLocation_gunjan[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_gunjan], encode_gunjantest)
print(results)
cv2.putText(gunjan_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )


cv2.imshow('main.img',gunjan)
cv2.imshow('test.img',gunjan_test)

cv2.waitKey(0)
cv2.destroyAllWindows()

