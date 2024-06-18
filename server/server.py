from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import imutils
from sklearn.preprocessing import LabelBinarizer
import joblib
from PIL import Image
from io import BytesIO
from flask_cors import cross_origin
import matplotlib.pyplot as plt

LB = joblib.load('./lb.pkl')

app = Flask(__name__)

# Carregar o modelo TensorFlow
model = tf.keras.models.load_model('model.h5')  # Substitua 'model.h5' pelo seu modelo

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_letters(image_bytes):
    letters = []
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_NEAREST)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,32,32,1)

        ypred = model.predict(thresh)
        ypred = LB.inverse_transform(ypred)
        [x] = ypred
        letters.append(x)

        filename = f'./imgs/letra_{x}.png'
        print(filename) 
        cv2.imwrite(filename,thresh)
    return letters, image

def np_array_to_base64(np_array):
    # Converte o array NumPy para uma imagem PIL
    image = Image.fromarray(np_array)

    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    # Converte a imagem para uma string base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str


def get_word(letter):
    word = "".join(letter)
    return word

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        image_base64 = file.read()
        letter,image = get_letters(image_base64)
        word = get_word(letter)
        base64_image = np_array_to_base64(image)
        return jsonify({'prediction': word, 'image': base64_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
