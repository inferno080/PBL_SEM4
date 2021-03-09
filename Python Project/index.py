import cv2

# Paths

TrafficPath = 'Resources/Inshot2.mp4'
NumberPlatePath = 'Resources/haarcascade_russian_plate_number.xml'
HelmetPath = 'Resources/Yes_boi.xml'

# Initialize values here

NumberPlateCascade = cv2.CascadeClassifier(NumberPlatePath)
Traffic1 = cv2.VideoCapture(TrafficPath)
HelmetCascade = cv2.CascadeClassifier(HelmetPath)

# Loops Inside the Input video

while True:
    success, img = Traffic1.read()
    img = cv2.resize(img, (700, 500))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = NumberPlateCascade.detectMultiScale(imgGray, 1.4, 8)        # Trial and Error for scale and min neighbours
    helmets = HelmetCascade.detectMultiScale(imgGray, 1.1, 6)
    helmets_x = 0
    helmets_y = 0
    plates_x = 0
    plates_y = 0
    for (x, y, w, h) in helmets:
        helmets_x = (2*x + w)/2
        helmets_y = (2*y + h)/2
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in plates:
        plates_x = (2*x + w)/2
        plates_y = (2*y + h)/2
        y_diff = abs(helmets_y - plates_y)
        x_diff = abs(helmets_x - plates_x)
        if y_diff < 700 and x_diff < 30:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Enter Ashmika's Code here
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

