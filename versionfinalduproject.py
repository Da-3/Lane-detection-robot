import cv2
import numpy as np
import utlis
import urllib.request
import requests
import httpx
import asyncio
from keras.models import load_model
import os
os.chdir('C:/Users/moukh/PycharmProjects/MODELAI/') # chemin de model AI
model = load_model('./training8/finalmodel.h5') # chemin de model AI


url = "http://192.168.137.92:8080/shot.jpg"
prediction=[]

classes = { 0:"speedlimit",
            1:"work",
            2:"stopsign"}

def find_area(contours):
    max_area = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area>max_area:
            biggest_contour = contours[i]
            max_area = area
    return max_area
def test_on_img(img):
    data=[]
    img = cv2.resize(img,(30, 30))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    data.append(np.array(img))
    X_test=np.array(data)
   # X_test = X_test.reshape(1,30,30,1)
    Y_pred = model.predict(X_test)
    return Y_pred

def returnRedness(img):
    yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    y,u,v=cv2.split(yuv)
    return v

def threshold(img,T=150):
    _,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
    return img

def findContour(img):
    contours = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours

def findBiggestContour(contours):
    m=0
    if len(contours)>0:
        c = [cv2.contourArea(i.astype(np.float32)) for i in contours]
        return contours[c.index(max(c))]
    else:
        return None

def boundaryBox(img,contours):
    x,y,w,h=cv2.boundingRect(contours)
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    sign=img[y:(y+h) , x:(x+w)]
    return img,sign

# Load the pre-trained Haar Cascade classifier for traffic signs
# Load your HAARCASCADE.XML after putting it in the same folder


curvelist = []
avgVal=10
data=[]
data_str =""
def send_data(message):
    IP_ADDRESS = "192.168.137.29"
    PORT = 80
    URL = f"http://{IP_ADDRESS}:{PORT}/"

    data = message.encode()

    async def send_data1():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(URL, data=data)
                response.raise_for_status()  # Raise an exception for non-2xx responses
                print(response.text)
            except httpx.ReadTimeout:
                print("Timeout occurred while reading the response.")
            except httpx.RequestError as err:
                print(f"An error occurred: {err}")

    if asyncio.get_event_loop().is_running():
        asyncio.create_task(send_data1())
    else:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(send_data1())
        loop.close()






def getlanecurve(img,display=2):
   imgCopy = img.copy()
   imgResult = img.copy()
   imgthres = utlis.thresholding(img)

   hT, wT, c = img.shape
   points = utlis.valTrackbars()
   imgWarp = utlis.wrapImg(imgthres,points,wT,hT)
   imgWarpPoints = utlis.drawPoints(imgCopy,points)

   middlePoint,imgHist = utlis.getHistogram(imgWarp,display=True,minPer=0.5,region=4)
   curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
   curveRaw = curveAveragePoint-middlePoint

   curvelist.append(curveRaw)
   if len(curvelist)>avgVal:
       curvelist.pop(0)
   curve=int(sum(curvelist)/len(curvelist))

   if display != 0:
       imgInvWarp = utlis.wrapImg(imgWarp, points, wT, hT, inv=True)
       imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
       imgLaneColor = np.zeros_like(img)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
       midY = 450
       cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
       cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                    (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
       #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
       #cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
   if display == 2:
       imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                            [imgHist, imgLaneColor, imgResult]))
       cv2.imshow('ImageStack', imgStacked)
       cv2.imshow('imgHist',imgHist)
   elif display == 1:
       cv2.imshow('Resutlt', imgResult)

       curve = curve/100
       if curve >1: curve==1
       if curve <-1 : curve ==-1



   #cv2.imshow('Thres',imgthres)
   #cv2.imshow('Warp',imgWarp)
   #cv2.imshow('Warp Points', imgWarpPoints)
   #cv2.imshow('Histogram', imgHist)
   return curve


def run1():
    initialTracbarsVals = [102, 121, 20, 214]
    utlis.initializeTrackbars(initialTracbarsVals)
    while True:
        # Getting Raw data

        RawData = requests.get(url, verify=False)

        # Convertting it to serilized one deminsion array
        One_D_Arry = np.array(bytearray(RawData.content), dtype=np.uint8)

        # converting One deminsion Array into opencv image matrxi, format using "imdecode" function

        frame = cv2.imdecode(One_D_Arry, -1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (480, 854))
        error = getlanecurve(frame, display=2)
        data_str = str(error) + ",7,7"
        #send_data(data_str)


        redness = returnRedness(frame)  # step 1 --> specify the redness of the image
        thresh = threshold(redness)
        contours = findContour(thresh)
        if contours is not None:
            big = findBiggestContour(contours)
            max_area=find_area(contours)
            if max_area > 3000:
                img, sign = boundaryBox(frame, big)
                prediction = test_on_img(sign)
                s = [i for i in prediction]
                a = np.array(s).argmax()
                print("a: ", a)
                data = [error, a + 1, 0]
                data_str = ','.join(str(value) for value in data)
                send_data(data_str)
                cv2.imshow('frame', frame)
                cv2.imshow('sign', sign)
                print("data_str: ", data_str)
                print("I am in if")
            else:
                cv2.imshow('frame', frame)
                data_str = str(error) + ",7,7"
                send_data(data_str)
                print("I am in else")
        else :
            print("no countours found in the image")

        print("error: ",error)
        # cv2.imshow('curve', curve)
        key = cv2.waitKey(1)
        if key == ord('q'):
         break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    run1()
