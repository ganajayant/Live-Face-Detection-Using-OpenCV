import time

import cv2 as cv

face_cascade = cv.CascadeClassifier(
    './model/haarcascade_frontalface_default.xml')


def detect(image, face_cascade):
    Image = image.copy()
    gray_image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
    box, detections = face_cascade.detectMultiScale2(
        gray_image, scaleFactor=1.1, minNeighbors=8)
    for x, y, w, h in box:
        cv.rectangle(Image, (x, y), (x+w, y+h), (0, 0, 0), 2)
    return Image


cap = cv.VideoCapture(0)
fps = 0
while True:
    start_time = time.time()
    _, frame = cap.read()
    frame = detect(frame, face_cascade)
    cv.putText(frame, 'FPS: {:.0f}'.format(fps), (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1.5,
               (255, 255, 255), 1)
    cv.putText(frame, 'Press q to exit', (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5,
               (255, 255, 255), 1)
    cv.imshow('Face Detection', frame)
    fps = 1/(time.time()-start_time)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
