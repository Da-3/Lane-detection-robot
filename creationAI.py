import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from keras.models import load_model
os.chdir('C:/Users/moukh/PycharmProjects/MODELAI/')
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []
labels = []

classes = 3

cur_path = os.getcwd()
cur_path
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.convert('L')
            image = image.resize((30,30))
            image = np.array(image)
            image.reshape(1, 30, 30, 1)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)
data = np.array(data)
labels = np.array(labels)
os.mkdir('training8')

np.save('./training8/data',data)
np.save('./training8/target',labels)
data=np.load('./training8/data.npy')
labels=np.load('./training8/target.npy')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(5,5), activation='relu', input_shape=(30,30,1)))
model.add(Conv2D(filters=100, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=100, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=100, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
# We have 3 classes that's why we have defined 3 in the dense
model.add(Dense(3, activation='softmax'))
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 50
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("./training8/modellakher8.h5")

os.chdir('C:/Users/moukh/PycharmProjects/MODELAI/')
model1 = load_model('./training8/modellakher8.h5')
classes = { 0:"speedlimit",
            1:"work",
            2:"stopsign"}
def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.convert('L')
    image = image.resize((30, 30))
    data.append(np.array(image))
    X_test=np.array(data)
    #X_test = X_test.reshape(1,30,30,3)
    Y_pred = model1.predict(X_test)
    return Y_pred
prediction = test_on_img(r'C:\Users\moukh\Downloads\3.29-1000x1000.png')
s = [i for i in prediction]
a = np.array(s).argmax()
print("Predicted traffic sign is: ", classes[a])
