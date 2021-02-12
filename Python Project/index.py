import cv2

# Paths

TrafficPath = 'Resources/Traffic 2.mp4'
NumberPlatePath = 'Resources/haarcascade_russian_plate_number.xml'
HelmetPath = 'Resources/helmetdetection.xml'

# Initialize values here

NumberPlateCascade = cv2.CascadeClassifier(NumberPlatePath)
Traffic1 = cv2.VideoCapture(TrafficPath)
HelmetCascade = cv2.CascadeClassifier(HelmetPath)

# Loops Inside the Input video

while True:
    success, img = Traffic1.read()
    img = cv2.resize(img, (500, 500))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = NumberPlateCascade.detectMultiScale(imgGray, 1.05, 1)        # Trial and Error for scale and min neighbours
    helmets = HelmetCascade.detectMultiScale(imgGray, 1.1, 1)
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in helmets:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

