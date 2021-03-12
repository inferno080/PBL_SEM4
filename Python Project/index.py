import cv2
import imutils as imutils
import numpy as np
import pytesseract
import re

# Paths
pytesseract.pytesseract.tesseract_cmd = 'Resources/Tesseract/tesseract.exe'
TrafficPath = 'Resources/inshot2.mp4'
NumberPlatePath = 'Resources/haarcascade_russian_plate_number.xml'
HelmetPath = 'Resources/helmetdetection_AARSY.xml'

# Initialize values here

NumberPlateCascade = cv2.CascadeClassifier(NumberPlatePath)
Traffic1 = cv2.VideoCapture(TrafficPath)
HelmetCascade = cv2.CascadeClassifier(HelmetPath)

number_plate = []
mylist = []
last_four = ""
final_list = []

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
    flag = 0
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
            flag = 1
        if flag == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 13, 15, 15)
            canny = cv2.Canny(gray, 30, 200)
            contours = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None

            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break
            if screenCnt is None:
                detected = 0
                pass
            else:
                detected = 1

            if detected == 1:
                #cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
                newimage = cv2.bitwise_and(img, img, mask=mask)

                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
                #cv2.imshow("crop", Cropped)

                text = pytesseract.image_to_string(Cropped, config="--psm 13")
                string_check = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
                if (string_check.search(text) != None):
                    res = ""
                    for character in text:
                        if character.isalnum():
                            res += character
                    text=res

                elif((text[0]).isnumeric()):
                    text=text[1:]

                elif(len(text)==0):
                    pass

                last_num = text[-4:]
                mylist.append(last_num)

                if (len(text) == 10):
                    number_plate.append(text)


    cv2.imshow("Video", img)
    if(len(mylist)!=0):
        last_four = max(set(mylist), key=mylist.count)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

    for s in number_plate:
        if (last_four==s[-4:]):
            final_list.append(s)

    if(len(final_list)==1):
        print("============================================")
        print("Detected number plate : ", final_list[0])
        print("============================================")
    else:
        pass
