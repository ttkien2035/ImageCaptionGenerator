from flask import Flask, render_template, jsonify, request
import numpy as np
from PIL import Image
from pickle import load
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('model_13.h5')
xception_model = Xception(include_top=False, pooling="avg")
def extract_features(filename, model):
  try:
    image = Image.open(filename)

  except:
    return []
  image = image.resize((299, 299))
  image = np.array(image)
  # for images that has 4 channels, we convert them into 3 channels
  if image.shape[2] == 4:
    image = image[..., :3]
  image = np.expand_dims(image, axis=0)
  image = image / 127.5
  image = image - 1.0
  feature = model.predict(image)
  return feature


def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None


def generate_desc(model, tokenizer, photo, max_length):
  in_text = 'start'
  for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    pred = model.predict([photo, sequence], verbose=0)
    pred = np.argmax(pred)
    word = word_for_id(pred, tokenizer)
    if word is None:
      break
    in_text += ' ' + word
    if word == 'end':
      break
  return in_text

@app.route("/", methods=['get'])
def index():
  if request.method == 'GET':
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
  if request.method == 'POST':
    file_to_upload = request.files['file']
    if file_to_upload:
      max_length = 32
      photo = extract_features(file_to_upload, xception_model)
      if photo == []: return jsonify({"message": "star Make sure choose image file and extension is correct end"})
      description = generate_desc(model, tokenizer, photo, max_length)
      return jsonify({"message": description})
