import cv2
import numpy as np
import requests


frameWidth = 640
frameHeight = 480
#cap = cv2.VideoCapture(1)
#cap.set(3, frameWidth)
#cap.set(4, frameHeight)
# chnage this url according yours
url = "http://192.168.0.100:8080/shot.jpg"


def empty(a):
    pass


cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)



while True:
    # Getting Raw data
    RawData = requests.get(url, verify=False)

    # Convertting it to serilized one deminsion array
    One_D_Arry = np.array(bytearray(RawData.content), dtype=np.uint8)

    # converting One deminsion Array into opencv image matrxi, format using "imdecode" function

    im = cv2.imdecode(One_D_Arry, -1)
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    im = cv2.resize(im, (480, 854))

    imgHsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    print(h_min)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(im, im, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([im, mask, result])
    cv2.imshow('Horizontal Stacking', hStack)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cv2.destroyAllWindows()