import cv2

#Trained DataSet
trainedDataset=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)
while True:
    sucess,frame=video.read()
    if sucess==True:
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = trainedDataset.detectMultiScale(gray_image)
        print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('video',frame)
        key=cv2.waitKey(1)
        if key ==81 or key==113:
            break
    else:
        print("video completed or frame nil")
        break