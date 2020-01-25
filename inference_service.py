import flask
from flask import Flask
from flask import request, jsonify
import logging
import tensorflow as tf
print(f'Using TensorFlow {tf.__version__}')
import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

input_elements = []
labels = []

with open("input-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        input_elements.append(article)
print(f'Labels count: {len(labels)}')
print(f'Input elements count: {len(input_elements)}')

train_size = int(len(input_elements) * training_portion)

train_input_elements = input_elements[0: train_size]
train_labels = labels[0: train_size]

validation_articles = input_elements[train_size:]
validation_labels = labels[train_size:]


# Create a basic model instance
model = tf.keras.models.load_model('training_1/my_model.h5')

# Show the model architecture
model.summary()

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_input_elements)
word_index = tokenizer.word_index

def classify(input_text):
    txt = [input_text]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    labels = ['sport', 'bussiness', 'politics', 'tech', 'entertainment']
    try:
        result = [str(labels[np.argmax(pred)]),str(np.max(pred))]
    except Exception:
        result = ["not certain", 0]
    return result

flask_app = Flask(__name__)

@flask_app.route('/inference', methods=['POST'])
def flask_post_endpoint():
    input_data = request.json
    output_data = classify(input_data['input_text'])
    logging.debug(input_data)

    return jsonify(isError=False,
                   message="Success",
                   statusCode=200,
                   data=({"class": output_data[0], "probability": output_data[1]})), 200

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=8888, debug=True)
