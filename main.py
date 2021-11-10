#import libraries
import os
import PIL
import tensorflow as tf
import pyscreenshot as ImageGrab
import cv2
import glob
import numpy as np
from tkinter import *
#from PIL import Image, ImageDraw, ImageGrab

#Load Model
from keras.models import load_model

model = load_model('my_model.h5')
print("Model Loaded successfully ")


def clear_window():
    global cv
    cv.delete("all")

def start_event(event):
    global lastx, lasty
    # <B1-Motion>
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    # do the canvas drawings
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def Recognize_Digit():
    global image_number
    predictions = []
    percentage = []
    # image number = 0
    filename = f'image_{image_number}.png'
    widget = cv

    # Get the widget coordinates
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Grab the image, crop it according to my requirement and saved it in png format
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    # read the image in color format
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    # Convert the image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Otsu thresholding
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # FindContour function helps in extracting the contours from the image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        'Get Bounding box and xtract ROI'
        x, y, w, h = cv2.boundingRect(cnt)
        # Create REctangles
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        # Extract the image ROI
        roi = th[y - top:y + h + bottom, x - left:x + w + right]
        # Resize ROI image to 28*28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # reshaping the image to support our model input
        img = img.reshape(1, 28, 28, 1)
        # Normalizing the image to support our model input
        img = img / 255.0
        # Its time to prdict the rsult
        pred = model.predict([img])[0]
        # numpy.argmax (input array) Returns the indices of the maimum values
        final_pred = np.argmax(pred)
        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
        # cv2.putText() method is used to draw a text string on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    cv2.imshow('Predictions', image)
    cv2.waitKey(0)
#Create a Main window first (named as  root)

root = Tk()
root.resizable(0,0)
root.title('Digit Recognition')

#Initialize few variables
lastx, lasty = None, None
image_number = 0

#Create Canvas for drwaing

cv = Canvas(root,width=640,height=480, bg = 'white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2 )

#Tkinter provides a powerful mechanism to let you deal with events yourself
cv.bind('<Button-1>', start_event)

#Add Buttons and Labels
btn_save = Button(text="Recognize Digit",command= Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1,padx=1)
button_clear = Button(text="Clear Widget", command= clear_window)
button_clear.grid(row=2,column=1,pady=1,padx=1)

#mainloop is used whn your application is ready to run
root.mainloop()
