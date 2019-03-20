from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
from copy import copy
from flask import Flask, request, jsonify, render_template
from model import Model
from vocabulary import Vocabulary
from caption_genarator import CaptionGenerator


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model", "/home/wojciech/Dokumenty/image_caption/etc/show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab", "/home/wojciech/Dokumenty/image_caption/etc/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("port", "5000", "Port of the server.")
tf.flags.DEFINE_string("host", "localhost", "Host of the server.")
tf.flags.DEFINE_integer("beam_size", 3, "Size of the beam.")
tf.flags.DEFINE_integer("max_caption_length", 20, "Maximum length of the generate caption.")

vocab = Vocabulary(FLAGS.vocab)
model = Model(model_path=FLAGS.model)
generator = CaptionGenerator(model=model, vocab=vocab, beam_size=FLAGS.beam_size, max_caption_length=FLAGS.max_caption_length)

logger = logging.getLogger(__name__)
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
@app.route('/api/image-caption/predict', methods=['GET','POST'])
def caption():
    if request.method == 'POST':
        file = request.files['image']
        file.save("static/images/" + file.filename)
        image = open("static/images/" + file.filename, 'rb').read()
        
        file_path = "images/" + file.filename
        print(file_path)
        captions = generator.beam_search(image)
        sentences = []
        for caption in captions:
            sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
            sentences.append((" ".join(sentence), np.exp(caption.logprob)))

        logger.info(sentences)
        print(sentences[0][0])
        print(sentences[1][0])
        print(sentences[2][0])
        
        #return 'ok'
        #return jsonify({"captions": sentences})
        return render_template('template.html', image=file_path, var1=sentences[0][0],var2=sentences[0][1],var3=sentences[1][0],var4=sentences[1][1],var5=sentences[2][0],var6=sentences[2][1],)
    return '''
        <!doctype html>
        <link class="jsbin" href="http://ajax.googleapis.com/ajax/libs/jqueryui/1/themes/base/jquery-ui.css" rel="stylesheet" type="text/css" />
        <script class="jsbin" src="http://ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
        <script class="jsbin" src="http://ajax.googleapis.com/ajax/libs/jqueryui/1.8.0/jquery-ui.min.js"></script>
        <meta charset=utf-8 />
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=image>
             <input type=submit value=Upload>
        </form>
        '''

if __name__ == '__main__':
    app.run(host=FLAGS.host, port=FLAGS.port)
