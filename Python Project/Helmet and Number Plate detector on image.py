import cv2

# Paths

TrafficPath = 'Resources/Both 2.PNG'
NumberPlatePath = 'Resources/haarcascade_russian_plate_number.xml'
HelmetPath = 'Resources/helmetdetection.xml'

# Initialize values here

NumberPlateCascade = cv2.CascadeClassifier(NumberPlatePath)
HelmetCascade = cv2.CascadeClassifier(HelmetPath)
Traffic1 = cv2.imread(TrafficPath)

# Code for single Image

# Traffic1 = cv2.resize(Traffic1, (500, 500))
Traffic1Gray = cv2.cvtColor(Traffic1, cv2.COLOR_BGR2GRAY)
plates = NumberPlateCascade.detectMultiScale(Traffic1Gray, 1.01, 1)       # Trial and Error for scale and min neighbours
helmets = HelmetCascade.detectMultiScale(Traffic1Gray, 1.1, 1)
for (x, y, w, h) in plates:
    cv2.rectangle(Traffic1, (x, y), (x + w, y + h), (255, 0, 0), 2)
for (x, y, w, h) in helmets:
    cv2.rectangle(Traffic1, (x, y), (x + w, y + h), (255, 0, 255), 2)
cv2.imshow("Output", Traffic1)
cv2.waitKey(0)

