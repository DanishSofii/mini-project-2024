from flask import Flask, request, render_template
import os
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load classification model
classification_model_path = 'D:/minip/minip/minip/notebooks/ocular_disease_vgg16_2.h5'
classification_model = tf.keras.models.load_model(classification_model_path)
class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(BytesIO(file.read()))
            img.save('static/uploaded_image.png')  # Save the image for displaying later

            # Preprocess and predict
            preprocessed_img = preprocess_image(img)
            prediction = classification_model.predict(preprocessed_img)[0]
            predicted_classes = list(zip(class_labels, prediction))
            sorted_predicted_classes = sorted(predicted_classes, key=lambda x: x[1], reverse=True)

            # Convert predictions to percentages and round
            sorted_predicted_classes = [(cls, round(prob * 100, 2)) for cls, prob in sorted_predicted_classes]

            # Render the same template with results
            return render_template('index.html', img_url='static/uploaded_image.png', predictions=sorted_predicted_classes)

    return render_template('index.html')

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    app.run(debug=True)
