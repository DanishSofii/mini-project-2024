import os
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import RRDBNet_arch as arch  # Assuming this defines the ESRGAN architecture
from flask import Flask, request, render_template, Response
# Load models outside request processing (assuming pre-loading)
classification_model = tf.keras.models.load_model('D:/minip/minip/minip/notebooks/ocular_disease_vgg16_2.h5')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
esrgan_model.load_state_dict(torch.load('D:/minip/minip/minip/notebooks/esrgan/RRDB_ESRGAN_x4.pth'), strict=True)
esrgan_model.eval()
esrgan_model = esrgan_model.to(device)

def preprocess_image(img):
  img = img.resize((224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0  # Rescale the image
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

class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    file = request.files['file']
    if file:
      img = Image.open(BytesIO(file.read()))

      # Super-resolve image (using optimized model/inference)
      enhanced_img = super_resolve_image(img)

      # Preprocess and predict
      preprocessed_img = preprocess_image(enhanced_img)
      prediction = classification_model.predict(preprocessed_img)[0]
      predicted_classes = list(zip(class_labels, prediction))
      sorted_predicted_classes = sorted(predicted_classes, key=lambda x: x[1], reverse=True)

      # Save the image (optional)
      img_path = 'D:/minip/minip/minip/notebooks/static/uploaded_image.png'
      enhanced_img.save(img_path)

      return Response(render_template('result_Three.html', predicted_classes=sorted_predicted_classes, img_path=img_path), content_type='text/html; charset=utf-8')
  return Response(render_template('index.html'), content_type='text/html; charset=utf-8')

if __name__ == '__main__':
  import sys
  import io
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
  app.run(debug=True)
