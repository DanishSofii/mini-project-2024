from flask import Flask, request, render_template, Response
import os
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import RRDBNet_arch as arch

app = Flask(__name__)


classification_model_path = 'D:/minip/notebooks/ocular_disease_vgg16_2.h5'
classification_model = tf.keras.models.load_model(classification_model_path)
class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']


esrgan_model_path = 'D:/minip/notebooks/esrgan/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu') 
esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
esrgan_model.load_state_dict(torch.load(esrgan_model_path), strict=True)
esrgan_model.eval()
esrgan_model = esrgan_model.to(device)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

def super_resolve_image(img):
    img = np.array(img) * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = esrgan_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(BytesIO(file.read()))
            enhanced_img = super_resolve_image(img)
            
           
            img_dir = 'D:/minip/notebooks/static'
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            img_path = os.path.join(img_dir, 'enhanced_image.png')
            enhanced_img.save(img_path)

            preprocessed_img = preprocess_image(img_path)
            prediction = classification_model.predict(preprocessed_img)[0]
            predicted_classes = list(zip(class_labels, prediction))
            sorted_predicted_classes = sorted(predicted_classes, key=lambda x: x[1], reverse=True)

            return render_template('index.html', img_url=img_path, predictions=sorted_predicted_classes)
    return render_template('index.html')

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    app.run(debug=True)
