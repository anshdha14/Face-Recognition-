import cv2
facecap=cv2.CascadeClassifier("/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/cv2/data/haarcascade_frontalcatface_extended.xml")
videocap=cv2.VideoCapture(0)
while True:
    ret, video_data = videocap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = facecap.detectMultiScale(
    col,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live", video_data)
    if cv2.waitKey(10) == ord("a"):
        break
videocap.release()
