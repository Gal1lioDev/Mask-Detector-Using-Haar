import cv2
import numpy as np 
import os
from keras.preprocessing.image import img_to_array
from keras import models,layers
import matplotlib.pyplot as plt
from keras.preprocessing import image
import imutils

model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.load_weights('FaceMaskModel.hdf5')

vidCap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('cascade.xml')
while True:
	(grabbed, frame) = vidCap.read()
	frame = imutils.resize(frame, width=250)
	frame = cv2.flip(frame, 1)
	rects = detector.detectMultiScale(frame, scaleFactor=1.1, 
										minNeighbors=3, 
										minSize=(30, 30),
	  							   		flags=cv2.CASCADE_SCALE_IMAGE)
	for x, y, w, h in rects:
		roi = frame[x:x+w+30, y:y+h+30]
		roi = cv2.resize(roi, (150, 150))
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
		roi = roi.astype('float32') / 255
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		result = model.predict(roi)[0][0]
		label = f'No Mask {int(100 - result*100)}%' if result > 0.5 else f'Mask {int(result*100)}%'
		color = (0, 0, 255) if result > 0.5 else (0, 255, 0)
		cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
		cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break


vidCap.release()
cv2.destroyAllWindows()

