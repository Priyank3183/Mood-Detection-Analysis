from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import os
from keras.models import load_model
from camera import Video

app=Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def start_page():
    return render_template('index.html')

@app.route('/home')
def home():
    # os.remove('static/result.jpg')
    return render_template('home.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/aboutus')
def about():
    return render_template('aboutus.html')

@app.route('/image', methods=['POST'])
def up_img():
    img = request.files['file']

    img.save('static/file.jpg')

    model=load_model('model_file_70.h5')
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

    frame=cv2.imread("static/file.jpg")
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 

    cv2.imwrite("static/result.jpg", frame)
    filename = "result.jpg"
    return render_template('image.html', filename = filename)

@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)