import cv2

face = cv2.CascadeClassifier('faces.xml')
eye = cv2.CascadeClassifier('eyes.xml')

webcam = cv2.VideoCapture(0)
webcam.set(3, 300)
webcam.set(4, 500)

while True:
    success, img = webcam.read()

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(grey, scaleFactor=2, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_grey = grey[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye.detectMultiScale(roi_grey, scaleFactor=2, minNeighbors=8)
        for (ex, ey, ew, eh) in eyes
            roi_color2 = roi_color[ey:ey +eh]
            roi_color2 = cv2.GaussianBlur(roi_color2, (51,51), 51)
            roi_color[ey:ey +eh] = roi_color2
    cv2.imshow('Res', img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
webcam.release()