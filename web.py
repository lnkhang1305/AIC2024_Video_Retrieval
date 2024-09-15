from flask import Flask, render_template, request, Response
from search import search_images_from_query, translate_to_EN, init_model
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')

@app.route('/video_retrieval', methods=['GET','POST'])
def get_data():
    return_data = search_images_from_query(translate_to_EN(request.form['query']), request.form['k'], model, index, client)
    return Response(json.dumps(return_data),  mimetype='application/json')

if __name__ == '__main__':
    model, index, client = init_model()
    app.run(host='0.0.0.0', debug = True)